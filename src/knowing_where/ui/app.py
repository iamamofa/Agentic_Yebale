"""Interactive context and evidence workbench built on the public pipeline."""

import hashlib
import json

import altair as alt
import pandas as pd
import streamlit as st

from . import services
from .styles import STYLE
from .flow import decision_flow_html
from .uncertainty import lab, uncertainty_panel
from .learning import start_here, simulator, algorithms, research_snapshot
from .live import live_models
from .sharing import sharing_panel
from ..routing import RoutingPolicy

PAGES = ('Start here', 'Live models', 'Routing simulator', 'Algorithm explorer', 'Overview',
         'Context studio', 'Experiments', 'Uncertainty lab', 'Predict', 'Research notebook')
LABELS = {'bayesian': 'Bayesian fusion', 'dempster_shafer': 'Dempster–Shafer',
          'equal_sources': 'Equal sources', 'majority': 'Source majority',
          'parametric_only': 'Parametric only'}
ERRORS = (ValueError, TypeError, KeyError, OSError, OverflowError)


def _header(eyebrow, title, description):
    st.caption(eyebrow.upper())
    st.title(title)
    st.write(description)


def _download(label, value, filename, key):
    st.download_button(label, services.download_bytes(value), file_name=filename,
                       mime='application/json', key=key)


def _input(label, sample, key, *, file_types, height=200):
    mode = st.radio(f'{label} source', ('Sample', 'Upload', 'Paste'), horizontal=True, key=f'{key}_mode')
    if mode == 'Sample':
        content = services.sample_text(sample)
        with st.expander(f'Preview {label.lower()}'):
            st.code(content[:8000] + ('\n…' if len(content) > 8000 else ''), language='json')
        return content
    if mode == 'Upload':
        file = st.file_uploader(label, type=file_types, key=f'{key}_upload')
        return file.getvalue() if file else None
    return st.text_area(label, key=f'{key}_text', height=height,
                        placeholder='Paste a valid JSON object or JSONL records…', max_chars=services.MAX_BYTES) or None


def _fingerprint(*values):
    digest = hashlib.sha256()
    for value in values:
        content = value if isinstance(value, bytes) else services.download_bytes(value)
        digest.update(len(content).to_bytes(8, 'big'))
        digest.update(content)
    return digest.hexdigest()


def _comparison(report, key):
    frame = pd.DataFrame([{'Method': LABELS[name], **result['overall']}
                          for name, result in report['comparisons'].items()]).set_index('Method')
    names = {'point_accuracy': 'Point accuracy', 'answer_rate': 'Answer rate',
             'mapped_set_coverage': 'Mapped set coverage',
             'canonical_answer_set_coverage': 'Actual answer set coverage',
             'candidate_recall': 'Candidate recall',
             'mean_normalized_entropy': 'Mean answer dispersion',
             'mean_top_two_margin': 'Mean top-two gap'}
    names = {name: label for name, label in names.items() if name in frame}
    selected = st.selectbox('Comparison metric', tuple(names), format_func=names.__getitem__, key=f'{key}_metric') or next(iter(names))
    chart = alt.Chart(frame.reset_index()).mark_bar(color='#168a80', size=24, cornerRadiusEnd=5).encode(
        y=alt.Y('Method:N', sort='-x', title=None),
        x=alt.X(f'{selected}:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0%'), title=names[selected]),
        tooltip=['Method:N', alt.Tooltip(f'{selected}:Q', format='.1%', title=names[selected])]
    ).properties(height=240).configure_view(stroke=None).configure_axis(gridColor='#e5ecef')
    st.altair_chart(chart, width='stretch')
    columns = ['point_accuracy', 'answer_rate', 'selective_accuracy', 'mapped_set_coverage',
               'canonical_answer_set_coverage', 'mean_set_size']
    display = frame[columns].rename(columns={**names, 'selective_accuracy': 'Selective accuracy',
                                            'mean_set_size': 'Mean set size'})
    formatting = {col: '{:.1%}' for col in display.columns if col != 'Mean set size'}
    formatting['Mean set size'] = '{:.2f}'
    st.dataframe(display.style.format(formatting, na_rep='—'), width='stretch')
    st.caption('Mapped coverage includes Ø when the reference answer is missing. '
               'Actual answer coverage and answer rate help assess practical usefulness.')


def _overview():
    _header('Knowing Where · Research workspace', 'Give every answer an evidence trail.',
            'Prepare context, compare source fusion, and inspect when a system chooses to abstain.')
    st.info('This workspace runs recorded answer paths. The bundled examples are synthetic; '
            'no live model or paid API is called.')
    report = services.sample_report()
    cols = st.columns(4)
    cols[0].metric('Source channels', '4', help='P parameters · W web/user · K knowledge · M memory')
    cols[1].metric('Comparison methods', len(report['comparisons']))
    cols[2].metric('Demo test questions', report['splits']['test']['count'])
    cols[3].metric('Paths per source', report['configuration']['paths_per_source'])
    st.subheader('From context to a decision')
    st.markdown(decision_flow_html(), unsafe_allow_html=True)
    left, right = st.columns([2, 1])
    with left:
        st.subheader('Explore the bundled experiment')
        _comparison(report, 'overview')
    with right:
        st.subheader('Choose a workspace')
        st.markdown('**New to the project?** Start with **Start here**, then try the '
                    '**Routing simulator** and **Algorithm explorer** in the sidebar.')
        st.markdown('**Context studio**\n\nEdit a request and inspect budgets, retained excerpts, '
                    'source conflicts, and routing decisions.\n\n'
                    '**Experiments**\n\nRun train/calibration/test comparisons and export a fitted system.\n\n'
                    '**Uncertainty lab**\n\nSeparate path ambiguity from source disagreement, and test source sensitivity.\n\n'
                    '**Predict**\n\nLoad a fitted system and inspect probabilities, prediction sets, and abstention reasons.')
        st.caption('Use the sidebar to switch workspaces. Results remain in this server session until you clear them or the session ends.')
    with st.expander('How to read the source labels'):
        st.table(pd.DataFrame({'Source': ['P', 'W', 'K', 'M'],
                               'Meaning': ['Model parameters', 'Web material and user turns',
                                           'Caller-designated verified knowledge', 'Interaction memory']}))


def _context():
    _header('01 · Context studio', 'Make room for what matters.',
            'Compile source-linked excerpts and compare models against context, cost, and latency limits.')
    text = _input('Context request', 'context', 'context_input', file_types=['json'])
    data = {}
    if text:
        try:
            data = services.json_input(text)
            if not isinstance(data, dict):
                raise ValueError('A context request must be a JSON object.')
            if not isinstance(data.get('question', ''), str):
                raise ValueError('Question must be text.')
            if not isinstance(data.get('policy', {}), dict):
                raise ValueError('Policy must be a JSON object.')
            RoutingPolicy(**data.get('policy', {}))
        except ERRORS as error:
            st.error(str(error))
            return
    with st.form('context_form'):
        question = st.text_area('Question', value=data.get('question', ''), height=110)
        columns = st.columns(3)
        mode = columns[0].selectbox('Routing preference', ('quality', 'economy'),
                    index=1 if data.get('policy', {}).get('selection_mode') == 'economy' else 0)
        max_cost = columns[1].number_input('Maximum estimated cost / request ($)', min_value=0.0,
                                          value=float(data.get('policy', {}).get('max_cost_usd', 0.02)
                                                      if data.get('policy', {}).get('max_cost_usd', 0.02) is not None else 0.02),
                                          step=0.001, format='%.3f')
        max_latency = columns[2].number_input('Maximum estimated latency (ms)', min_value=0,
                                              value=int(data.get('policy', {}).get('max_latency_ms', 1500)
                                                        if data.get('policy', {}).get('max_latency_ms', 1500) is not None else 1500), step=100)
        submitted = st.form_submit_button('Compile context', type='primary', disabled=not text)
    signature = _fingerprint(text, question, mode, max_cost, max_latency)
    if submitted:
        st.session_state.pop('context_result', None)
        try:
            with st.spinner('Compiling per-model packets…'):
                report = services.compile_context(text, question=question, selection_mode=mode,
                                                   max_cost=max_cost, max_latency=max_latency)
            st.session_state['context_result'] = report
            st.session_state['context_signature'] = signature
        except ERRORS as error:
            st.error(f'Context could not be compiled: {error}')
    report = st.session_state.get('context_result')
    if not report:
        st.caption('Choose a sample or supply a request, then compile to inspect the result.')
        return
    if signature != st.session_state.get('context_signature'):
        st.info('Inputs changed. Showing the last completed result; compile again to apply your changes.')
    st.divider()
    st.subheader('Routing decision')
    (st.success if report['selected_model_id'] else st.warning)(report['decision_reason'])
    _download('Download context report', report, 'context-report.json', 'download_context')
    if report['rejected']:
        with st.expander('Models that did not qualify'):
            st.table(pd.DataFrame([{'Model': model, 'Reason': reason} for model, reason in report['rejected'].items()]))
    if not report['candidates']:
        return
    candidate_id = st.selectbox('Inspect a model', [c['model_id'] for c in report['candidates']])
    candidate = next(c for c in report['candidates'] if c['model_id'] == candidate_id)
    audit = candidate['audit']
    cols = st.columns(3)
    cols[0].metric('Input used', f"{audit['used_units']:,} / {audit['budget_units']:,}")
    cols[1].metric('Query-term coverage', f"{candidate['evaluation']['query_term_coverage']:.0%}")
    cols[2].metric('Declared suitability', f"{candidate['suitability']:.2f}")
    st.progress(min(1.0, audit['used_units'] / audit['budget_units']))
    st.caption('Budget units follow the model profile. The bundled profiles use UTF-8 bytes. '
               'Suitability and lexical coverage are diagnostics, not correctness probabilities.')
    excerpts, conflicts, omissions, raw = st.tabs(('Retained evidence', 'Claim conflicts', 'Omissions & budget sweep', 'Exact payload'))
    with excerpts:
        sources = {d['id']: d['source'] for d in candidate['context']['documents']}
        for index, excerpt in enumerate(candidate['context']['excerpts']):
            with st.expander(f"{sources[excerpt['document_id']]} · {excerpt['document_id']} · "
                             f"characters {excerpt['start']}–{excerpt['end']}", expanded=index < 2):
                st.code(excerpt['text'], language=None)
        with st.expander('Source metadata and provenance'):
            st.json(candidate['context']['documents'])
    with conflicts:
        if candidate['context']['conflicts']:
            st.json(candidate['context']['conflicts'])
        else:
            st.success('No explicitly declared claim conflicts in this packet.')
    with omissions:
        st.json(audit.get('omissions', []))
        st.dataframe(pd.DataFrame([{k: row.get(k) for k in ('budget_units', 'used_units', 'status', 'reason')}
                                    for row in report.get('budget_sweep', [])]), width='stretch')
    with raw:
        st.code(candidate['payload'], language='json')
        st.download_button('Download exact payload', candidate['payload'], file_name=f'{candidate_id}-payload.json',
                           mime='application/json')


def _experiment_results(report):
    st.divider()
    st.subheader('Experiment results')
    cols = st.columns(4)
    for col, name in zip(cols[:3], ('train', 'calibration', 'test')):
        col.metric(f'{name.title()} questions', report['splits'][name]['count'])
    cols[3].metric('Generator attempts', report['generation']['attempted_calls'])
    _comparison(report, 'experiment')
    compare, inspect, diagnostics, export = st.tabs(('Group results', 'Evidence inspector', 'Run diagnostics', 'Export'))
    with compare:
        method = st.selectbox('Group comparison method', tuple(report['comparisons']), format_func=LABELS.__getitem__)
        group_frame = pd.DataFrame(report['comparisons'][method]['groups']).T
        st.dataframe(group_frame[['count', 'point_accuracy', 'answer_rate', 'selective_accuracy',
                                   'mapped_set_coverage', 'canonical_answer_set_coverage', 'mean_set_size']],
                     width='stretch')
    with inspect:
        records = report['test_predictions']['bayesian']
        id_ = st.selectbox('Test question', [r['id'] for r in records])
        st.caption('Reference labels below are held-out evaluation data; they are not generation inputs.')
        record = next(r for r in records if r['id'] == id_)
        question = st.session_state.get('experiment_questions', {}).get(id_)
        if question:
            st.text(question)
        st.text(f"Reference answer: {record['gold']} · Group: {record['group']}")
        rows = []
        for name, predictions in report['test_predictions'].items():
            r = next(p for p in predictions if p['id'] == id_)
            rows.append({'Method': LABELS[name], 'Answer': r['decision']['answer'] or 'Abstain',
                         'Reason': r['decision']['reason'], 'Set': ', '.join(r['prediction_set'])})
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
        audit = report['test_audit'][id_]
        if audit.get('source_estimates'):
            st.subheader('Source estimates')
            st.dataframe(pd.DataFrame(audit['source_estimates']).T, width='stretch')
        uncertainty_panel(record['probabilities'], audit, summary=record.get('uncertainty'), key='experiment')
        with st.expander('Routing, source paths, and conflict diagnostics'):
            st.json(audit)
    with diagnostics:
        st.json(report.get('quality_summary', {}))
        st.json(report['generation'])
        st.json(report['calibration'])
    with export:
        _download('Download full experiment', report, 'experiment-report.json', 'experiment_download')
        _download('Download fitted system', st.session_state['fitted_system'], 'system.json', 'system_download')
        st.caption('The fitted system contains coefficients and frozen group thresholds, without reference labels or test predictions.')


def _experiments():
    _header('02 · Experiments', 'Compare decisions, not just scores.',
            'Fit on training data, calibrate on a separate partition, and compare five methods on held-out questions.')
    left, right = st.columns([3, 2])
    with left:
        dataset = _input('Dataset', 'dataset', 'experiment_dataset', file_types=['jsonl', 'txt'])
    with right:
        models = _input('Model profiles', 'models', 'experiment_models', file_types=['json'])
    with st.form('experiment_form'):
        cols = st.columns(4)
        seed = cols[0].number_input('Split seed', value=42, step=1)
        alpha = cols[1].number_input('Miscoverage α', min_value=0.01, max_value=0.99, value=0.1, step=0.01,
                                     help='Target error rate for prediction sets. Smaller values often retain larger sets.')
        paths = cols[2].number_input('Paths per source', min_value=1, max_value=100, value=10, step=1)
        max_calls = cols[3].number_input('Total call budget', min_value=0, max_value=200000, value=4000, step=100)
        mode = st.selectbox('Routing preference', ('quality', 'economy'), key='experiment_routing')
        acknowledged = st.checkbox('Use the corpus-absence independence approximation', value=False)
        st.caption('Bayesian fusion uses the product of source ghost masses for its prior. '
                   'This is an explicit modeling assumption. The example uses recorded paths, not live model calls.')
        run = st.form_submit_button('Run experiment', type='primary', disabled=not(dataset and models))
    signature = _fingerprint(dataset, models, seed, alpha, paths, max_calls, mode)
    if run:
        if not acknowledged:
            st.warning('Select the independence approximation to run this experiment.')
        else:
            for key in ('experiment_report', 'fitted_system'):
                st.session_state.pop(key, None)
            try:
                with st.spinner('Routing, fitting, calibrating, and evaluating…'):
                    report, artifact = services.experiment(dataset, models, seed=int(seed), alpha=alpha,
                            paths=int(paths), max_calls=int(max_calls), policy={'selection_mode': mode})
                st.session_state['experiment_report'] = report
                st.session_state['fitted_system'] = artifact
                st.session_state['experiment_signature'] = signature
                st.session_state['experiment_questions'] = {r.id: r.question for r in
                                                              services.jsonl_input(dataset, labeled=True)}
                st.success('Experiment complete. The fitted system is also available in Predict.')
            except ERRORS as error:
                st.error(f'Experiment could not run: {error}')
    if st.session_state.get('experiment_report'):
        if signature != st.session_state.get('experiment_signature'):
            st.info('Inputs changed. Showing the last completed experiment; run again to apply your changes.')
        _experiment_results(st.session_state['experiment_report'])
    else:
        st.caption('Run the sample to reproduce the full pipeline, or upload prepared JSONL data and model cards.')


def _predict():
    _header('03 · Predict', 'See the answer—and the uncertainty.',
            'Use saved coefficients and calibration thresholds to make decisions without reference answers.')
    sources = ['Bundled system', 'Upload system', 'Paste system']
    if st.session_state.get('fitted_system'):
        sources.insert(0, 'Latest experiment')
    source = st.selectbox('Fitted system', sources)
    artifact = None
    if source == 'Latest experiment':
        artifact = json.dumps(st.session_state['fitted_system'])
    elif source == 'Bundled system':
        artifact = services.sample_text('system')
    elif source == 'Upload system':
        file = st.file_uploader('Fitted system JSON', type=['json'])
        artifact = file.getvalue() if file else None
    else:
        artifact = st.text_area('Fitted system JSON', height=180) or None
    queries = _input('Prediction queries', 'queries', 'prediction_queries', file_types=['jsonl', 'txt'])
    max_calls = st.number_input('Prediction call budget', min_value=0, max_value=200000, value=120, step=40)
    st.caption('Queries must exclude gold, corpus_present, and split_group. Replay requires recorded source paths.')
    signature = _fingerprint(artifact, queries, max_calls)
    if st.button('Run predictions', type='primary', disabled=not(artifact and queries)):
        st.session_state.pop('prediction_result', None)
        try:
            with st.spinner('Loading frozen system and scoring queries…'):
                result = services.predict(artifact, queries, max_calls=int(max_calls))
            st.session_state['prediction_result'] = result
            st.session_state['prediction_signature'] = signature
            st.session_state['prediction_questions'] = {q.id: q.question for q in
                                                          services.jsonl_input(queries, labeled=False)}
        except ERRORS as error:
            st.error(f'Predictions could not run: {error}')
    result = st.session_state.get('prediction_result')
    if not result:
        return
    if signature != st.session_state.get('prediction_signature'):
        st.info('Inputs changed. Showing the last completed predictions; run again to apply your changes.')
    st.divider()
    _download('Download predictions', result, 'predictions.json', 'predictions_download')
    st.caption(f"{len(result['predictions'])} questions · {result['generation']['attempted_calls']} generator attempts")
    id_ = st.selectbox('Inspect question', [r['id'] for r in result['predictions']])
    record = next(r for r in result['predictions'] if r['id'] == id_)
    method = st.selectbox('Scoring method', tuple(record['methods']), format_func=LABELS.__getitem__)
    question = st.session_state.get('prediction_questions', {}).get(id_)
    if question:
        st.text(question)
    prediction = record['methods'][method]
    decision = prediction['decision']
    if decision['abstained']:
        st.warning(f"Abstained · {decision['reason'].replace('_', ' ')}")
    else:
        st.success(f"Answer: {decision['answer']}")
    st.write('**Prediction set:**', ', '.join(prediction['prediction_set']) or 'Empty set')
    uncertainty_panel(prediction['probabilities'], record['audit'],
                      summary=prediction.get('uncertainty'), key='predict', calibration=prediction['calibration'])
    st.caption('Ø is the outside option. Probability scores and set coverage do not guarantee that a singleton answer is correct.')
    with st.expander('Calibration and evidence audit'):
        st.json(prediction['calibration'])
        st.json(record['audit'])


def main():
    st.set_page_config(page_title='Knowing Where · Evidence Workbench', page_icon=':material/explore:', layout='wide')
    st.html(STYLE)
    with st.sidebar:
        st.markdown('### :material/explore: Knowing Where')
        st.caption('Find the source. Follow the evidence.')
        st.caption('LEARN → SIMULATE → EXPERIMENT')
        page = st.radio('Workspace', PAGES, label_visibility='collapsed', key='workspace') or 'Start here'
        st.divider()
        st.markdown('<div class="source-chips"><span>P · Parameters</span><span>W · Web / user</span>'
                    '<span>K · Knowledge</span><span>M · Memory</span></div>', unsafe_allow_html=True)
        st.divider()
        st.caption('Local workspace · Replay & Hugging Face live inference')
        sharing_panel()
        if st.button('Clear session results'):
            for key in ('context_result', 'experiment_report', 'fitted_system', 'prediction_result',
                        'context_signature', 'experiment_signature', 'prediction_signature',
                        'experiment_questions', 'prediction_questions',
                        'hf_live_result', 'hf_live_signature'):
                st.session_state.pop(key, None)
            st.rerun()
    {'Start here': start_here, 'Live models': live_models,
     'Routing simulator': simulator, 'Algorithm explorer': algorithms,
     'Research notebook': research_snapshot, 'Overview': _overview, 'Context studio': _context,
     'Experiments': _experiments, 'Uncertainty lab': lab, 'Predict': _predict}[page]()
