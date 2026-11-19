"""Readable uncertainty panels and an interactive source-disagreement laboratory."""

from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from ..evidence.uncertainty import predictive_uncertainty, risk_coverage_curve, source_uncertainty
from ..types import OUTSIDE
from . import services

METHOD_NAMES = {'bayesian': 'Bayesian fusion', 'dempster_shafer': 'Dempster–Shafer',
                'equal_sources': 'Equal sources', 'majority': 'Source majority', 'parametric_only': 'Parametric only'}


def _chart(chart):
    st.altair_chart(chart.configure_view(stroke=None).configure_axis(
        labelColor='#627388', titleColor='#37485e', gridColor='#edf1f5', titleFontWeight=500),
        use_container_width=True)


def _ring(uncertainty):
    value = uncertainty['normalized_entropy']
    if value is None:
        st.info('Dispersion unavailable: ' + uncertainty['reason'].replace('_', ' '))
        return
    percent = round(value * 100)
    st.markdown(f'''<div class="uncertainty-orbit" role="img"
        aria-label="Normalized answer dispersion: {percent} percent"
        style="--orbit-angle:{value * 360:.3f}deg">
        <div class="orbit-center"><span class="orbit-number">{percent}<small>%</small></span>
        <span class="orbit-label">ANSWER DISPERSION</span></div></div>''', unsafe_allow_html=True)
    st.caption('Normalized entropy · 0% = concentrated · 100% = uniform across this frame. '
               'This is not the chance of being wrong.')


def probability_chart(probabilities, *, height=230):
    data = pd.DataFrame([{'Candidate': a, 'Score': p, 'Kind': 'Outside' if a == OUTSIDE else 'Answer'}
                         for a, p in probabilities.items()])
    chart = alt.Chart(data).mark_bar(cornerRadiusEnd=5, size=23).encode(
        y=alt.Y('Candidate:N', sort='-x', title=None, axis=alt.Axis(labelLimit=200)),
        x=alt.X('Score:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0%'), title='Probability score'),
        color=alt.Color('Kind:N', scale=alt.Scale(domain=['Answer', 'Outside'], range=['#168a80', '#d69a42']), legend=None),
        tooltip=['Candidate:N', alt.Tooltip('Score:Q', format='.1%')]).properties(height=height)
    _chart(chart)


def _sources_panel(summary):
    if not summary.get('available'):
        st.info('Source-path uncertainty is unavailable for this question.')
        return
    left, right = st.columns([3, 2])
    with left:
        data = pd.DataFrame([
            {'Component': 'Within-source ambiguity', 'Bits': summary['within_source_entropy_bits'], 'Mixture': 'Source mixture'},
            {'Component': 'Between-source disagreement', 'Bits': summary['between_source_disagreement_bits'], 'Mixture': 'Source mixture'}])
        chart = alt.Chart(data).mark_bar(size=42, cornerRadiusEnd=4).encode(
            y=alt.Y('Mixture:N', title=None, axis=None),
            x=alt.X('Bits:Q', title='Entropy of the equal-source mixture (bits)', stack='zero'),
            color=alt.Color('Component:N', scale=alt.Scale(
                domain=['Within-source ambiguity', 'Between-source disagreement'], range=['#168a80', '#e4a34c']),
                legend=alt.Legend(orient='bottom', title=None, labelLimit=260)),
            order=alt.Order('Component:N', sort='descending'),
            tooltip=['Component:N', alt.Tooltip('Bits:Q', format='.4f')]).properties(height=140)
        _chart(chart)
    with right:
        st.metric('Between-source disagreement', f"{summary['normalized_disagreement']:.1%}",
                  help='Jensen–Shannon gap divided by log₂(min(source count, candidate count)). '
                       'This describes differing source distributions, not correctness.')
        st.caption('The two components add to source-mixture entropy. They do not decompose '
                   'the selected fusion method’s entropy, or identify true epistemic and aleatoric uncertainty.')
    data = pd.DataFrame([{'Source': source, 'Candidate': candidate,
                          'Vote share': row['probabilities'].get(candidate, 0)}
                         for source, row in summary['sources'].items()
                         for candidate in summary['mixture_probabilities']])
    chart = alt.Chart(data).mark_rect(cornerRadius=4, stroke='#ffffff', strokeWidth=3).encode(
        x=alt.X('Candidate:N', title=None, axis=alt.Axis(labelAngle=0, labelLimit=160)),
        y=alt.Y('Source:N', title=None, sort=['P', 'W', 'K', 'M']),
        color=alt.Color('Vote share:Q', scale=alt.Scale(domain=[0, 1], range=['#edf4f5', '#0d756d']),
                        legend=alt.Legend(format='.0%', title='Path votes')),
        tooltip=['Source:N', 'Candidate:N', alt.Tooltip('Vote share:Q', format='.1%')]).properties(height=160)
    _chart(chart)
    with st.expander('Source sample sizes and ambiguity'):
        st.dataframe(pd.DataFrame([{'Source': source, 'Recorded paths': row.get('path_count'),
                                   'Entropy (bits)': row['entropy_bits'],
                                   'Variation ratio': row['variation_ratio'],
                                   'Leading vote share': row['top_probability']}
                                  for source, row in summary['sources'].items()]),
                     hide_index=True, width='stretch')
    st.caption('Sources remain separate channels. Ten agreeing paths are not ten independent pieces of evidence.')


def _sensitivity_panel(sensitivity):
    if not sensitivity.get('available'):
        st.info('Source sensitivity needs a valid Bayesian posterior.')
        return
    data = pd.DataFrame([{'Removed source': source, 'Distribution shift': row['total_variation'],
                          'Top candidate changed': row['top_candidate_changed'],
                          'New top candidate': row['top_candidate_without_source']}
                         for source, row in sensitivity['sources'].items() if row.get('available')])
    if data.empty:
        st.info('No valid source-removal comparisons are available.')
        return
    chart = alt.Chart(data).mark_bar(cornerRadiusEnd=4, size=24).encode(
        y=alt.Y('Removed source:N', title=None, sort='-x'),
        x=alt.X('Distribution shift:Q', title='Total variation from the original posterior',
                scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0%')),
        color=alt.Color('Top candidate changed:N', scale=alt.Scale(domain=[False, True], range=['#168a80', '#d69a42']),
                        legend=alt.Legend(title='Top answer changed', orient='bottom')),
        tooltip=['Removed source:N', alt.Tooltip('Distribution shift:Q', format='.1%'),
                 'Top candidate changed:N', 'New top candidate:N']).properties(height=190)
    _chart(chart)
    st.caption('Bayesian sensitivity only: remove one source, keep the candidate frame fixed, '
               'and recompute the joint ghost prior. A shift is not causal importance or proof that a source is trustworthy.')


def uncertainty_panel(probabilities, audit, *, summary=None, key='uncertainty', calibration=None):
    summary = summary or predictive_uncertainty(probabilities)
    st.subheader('Uncertainty fingerprint')
    left, right = st.columns([1, 2])
    with left:
        _ring(summary)
    with right:
        if summary['available']:
            columns = st.columns(3)
            columns[0].metric('Leading score', f"{summary['top_probability']:.1%}")
            columns[1].metric('Top-two gap', f"{summary['top_two_margin'] * 100:.1f} pp")
            columns[2].metric('Outside score', f"{summary['outside_probability']:.1%}")
            st.caption(f"{summary['entropy_bits']:.3f} bits · "
                       f"{summary['effective_candidates']:.2f} effective candidates · "
                       f"{summary['candidate_count']} entries in the declared frame, including zero-score entries.")
            probability_chart(probabilities)
        else:
            st.warning('The outside-only fallback is an abstention placeholder. No valid posterior uncertainty can be quantified for this result.')
    if calibration and calibration.get('full_frame'):
        st.info('Calibration retains the full frame for this group. Concentrated probabilities alone '
                'do not justify answering when calibration is sparse or the group is unseen.')
    source_tab, influence_tab, definitions = st.tabs(('Why are the paths uncertain?', 'Which source moves the result?', 'How to read this'))
    with source_tab:
        _sources_panel(audit.get('source_uncertainty', {}))
    with influence_tab:
        _sensitivity_panel(audit.get('source_sensitivity', {}))
    with definitions:
        st.markdown('**Answer dispersion** is normalized Shannon entropy of the selected score distribution. '
                    '**Within-source ambiguity** means one channel produces different answers. '
                    '**Between-source disagreement** means the channels’ distributions differ. '
                    '**Outside score** belongs to Ø, representing truth outside the candidate frame under the scoring model.')
        st.caption('Agreement can be confidently wrong. These descriptive measures do not estimate parameter uncertainty, '
                   'prove evidence sufficiency, or replace held-out calibration. Their calculations use no reference labels.')
        st.json(summary)


def _risk_review(records, key):
    points = risk_coverage_curve(records)
    st.subheader('Review fewer questions. See what changes.')
    st.caption('Exploratory held-out point-prediction view. This does not change saved conformal decisions '
               'or establish a new calibrated threshold.')
    if not points:
        st.info('There are no valid non-outside point predictions in this subset.')
        return
    ceiling = st.slider('Exploratory entropy ceiling', 0.0, 1.0, 0.5, 0.05, key=f'{key}_ceiling')
    included = [r for r in records if r.get('routed', True) and r['uncertainty']['available'] and
                r['uncertainty']['top_candidate'] != OUTSIDE and r['uncertainty']['normalized_entropy'] <= ceiling]
    columns = st.columns(2)
    columns[0].metric('Questions in review subset', f'{len(included)} / {len(records)}')
    errors = sum(r['uncertainty']['top_candidate'] != r['gold'] for r in included)
    columns[1].metric('Observed point-answer error', f'{errors / len(included):.1%}' if included else '—')
    frame = pd.DataFrame(points)
    line = alt.Chart(frame).mark_line(point=True, color='#168a80', strokeWidth=2.5).encode(
        x=alt.X('coverage:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0%'), title='Fraction of test questions retained'),
        y=alt.Y('error_rate:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0%'), title='Observed canonical-answer error'),
        tooltip=[alt.Tooltip('coverage:Q', format='.1%'), alt.Tooltip('error_rate:Q', format='.1%'),
                 'reviewed_count:Q', alt.Tooltip('entropy_threshold:Q', format='.3f')]).properties(height=240)
    _chart(line)


def _controlled():
    st.markdown('### Same uncertainty. Different reasons.')
    st.write('Compare agreement, ambiguity within every source, and disagreement between sources. '
             'Move the four controls to build your own case.')
    presets = {'Sources agree': [1.0, 1.0, 1.0, 1.0],
               'Each source is split': [0.5, 0.5, 0.5, 0.5],
               'Sources disagree': [1.0, 1.0, 0.0, 0.0]}
    preset = st.selectbox('Starting scenario', tuple(presets), key='uncertainty_preset')
    columns = st.columns(4)
    distributions = {}
    for column, source, default in zip(columns, ('P', 'W', 'K', 'M'), presets[preset]):
        value = column.slider(f'{source} · vote share for A', 0.0, 1.0, default, 0.1,
                              key=f'scenario_{preset}_{source}')
        distributions[source] = {'Answer A': value, 'Answer B': 1 - value}
    sources = source_uncertainty(distributions)
    probabilities = {**sources['mixture_probabilities'], OUTSIDE: 0.0}
    st.caption('Controlled illustration: equal-source mixture, not a fitted model or a calibrated decision. '
               'The fixed displayed frame is {A, B, Ø}; no additional model calls are made.')
    uncertainty_panel(probabilities, {'source_uncertainty': sources}, key='controlled')


def lab():
    st.caption('04 · UNCERTAINTY LAB')
    st.title('Know what you don’t know.')
    st.write('Separate answer ambiguity from source disagreement—and inspect what makes a decision fragile.')
    mode = st.radio('Explore', ('Run results', 'Controlled examples'), horizontal=True, key='uncertainty_mode')
    if mode == 'Controlled examples':
        _controlled()
        return
    sources = ['Bundled experiment']
    if st.session_state.get('experiment_report'):
        sources.insert(0, 'Latest experiment')
    controls = st.columns([1.2, 1.2, 1, 1.25])
    source = controls[0].selectbox('Experiment to inspect', sources, key='uncertainty_report')
    report = st.session_state['experiment_report'] if source == 'Latest experiment' else services.sample_report()
    method = controls[1].selectbox('Uncertainty scoring method', tuple(report['comparisons']),
                          format_func=METHOD_NAMES.get, key='uncertainty_method')
    records = report['test_predictions'][method]
    group = controls[2].selectbox('Question group', ['All groups', *sorted({r['group'] for r in records})], key='uncertainty_group')
    records = [{**r, 'uncertainty': r.get('uncertainty') or predictive_uncertainty(r['probabilities'],
                unavailable_reason=None if r.get('routed', True) else 'no_feasible_model')}
               for r in records if group == 'All groups' or r['group'] == group]
    id_ = controls[3].selectbox('Question to inspect', [r['id'] for r in records], key='uncertainty_question')
    row = next(r for r in records if r['id'] == id_)
    summary = row.get('uncertainty') or predictive_uncertainty(row['probabilities'])
    calibration = report['calibration'][method].get(row['group'], {'full_frame': True})
    decision = row['decision']
    status = 'Abstained' if decision['abstained'] else 'Answered'
    st.markdown(f'<div class="decision-strip"><span>{status}</span><strong>{escape(decision["reason"].replace("_", " "))}</strong>'
                f'<span>{escape(row["group"])}</span></div>', unsafe_allow_html=True)
    uncertainty_panel(row['probabilities'], report['test_audit'][id_], summary=summary, key='lab', calibration=calibration)
    st.divider()
    _risk_review(records, 'lab')
