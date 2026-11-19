"""Beginner workspaces and an explicitly illustrative source-first policy lab.

No benchmark outcomes, trained systems, or live provider calls are generated here.
"""

import json
from math import isfinite

import altair as alt
import pandas as pd
import streamlit as st

from .flow import decision_flow_html

SOURCES = {'P': 'Model knowledge', 'W': 'Web', 'K': 'Knowledge base', 'M': 'Memory'}
COLORS = ['#168a80', '#d69a42', '#7184b1']
SCENARIOS = {
    'A fact in company documents': ('What is the company travel reimbursement limit?',
                                  [(0.75, .50), (.70, .65), (.10, .90), (.45, .65)]),
    'A recent public event': ('Who won the latest tournament?',
                            [(.85, .60), (.15, .85), (.70, .80), (.90, .55)]),
    'A previous conversation': ('Which presentation style did I choose last time?',
                               [(.95, .60), (.98, .80), (.80, .85), (.10, .90)]),
    'Missing everywhere': ('What is the unpublished access code?',
                           [(.95, .75), (.98, .80), (.95, .90), (.97, .80)]),
}


def route_illustration(estimates, costs, penalty):
    """Rank query-side estimates, before generating any answer paths.

    Costs and penalty are dimensionless teaching controls, not prices.
    Ties use source ID for reproducibility. A nonpositive utility skips retrieval.
    """
    if not estimates or set(estimates) != set(costs):
        raise ValueError('Provide matching source estimates and costs.')
    if not isfinite(penalty) or penalty < 0:
        raise ValueError('Penalty must be finite and nonnegative.')
    rows = []
    for source, (ghost, reliability) in estimates.items():
        if any(not isfinite(v) or not 0 <= v <= 1 for v in (ghost, reliability, costs[source])):
            raise ValueError('Estimates and cost units must be in [0, 1].')
        success = (1 - ghost) * reliability
        rows.append({'Source': source, 'Ghost mass': ghost, 'Reliability': reliability,
                     'Useful retrieval': success, 'Cost units': costs[source],
                     'Utility': success - penalty * costs[source]})
    rows.sort(key=lambda row: (-row['Utility'], row['Source']))
    selected = rows[0]['Source'] if rows[0]['Utility'] > 0 else None
    return selected, rows


def outcome_masses(ghost, reliability):
    """Expected categories under the teaching model; no random benchmark run."""
    return {'Found': (1 - ghost) * reliability,
            'Present, missed': (1 - ghost) * (1 - reliability), 'Absent': ghost}


def _go(page):
    st.session_state['workspace'] = page


def _link(label, page, key):
    st.button(label, key=key, on_click=_go, args=(page,), use_container_width=True)


def start_here():
    st.caption('START HERE · A FIVE-MINUTE TOUR')
    st.title('A better answer starts with knowing where to look.')
    st.write('Imagine asking a librarian a question. A better librarian can find a hidden fact, '
             'but cannot find a fact in a book that never contained it. Our system measures those '
             'two problems separately, then uses that distinction to guide a decision.')
    st.info('The guided examples and simulations need no API key. To ask a real hosted model, '
            'open Live models and connect a Hugging Face token.')
    _link('Open Live models', 'Live models', 'tour_live')
    a, b = st.columns(2)
    with a:
        st.subheader('Is the answer there?')
        st.write('**Ghost mass (g)** estimates how likely the answer is to be absent from a source. '
                 'Lower is better. A failed search alone cannot establish absence.')
    with b:
        st.subheader('Can the agent find it?')
        st.write('**Reliability (r)** estimates how often the agent succeeds **when the answer is present**. '
                 'Higher is better. It is different from overall accuracy.')
    st.markdown('**Example:** If 20 of 100 questions have no answer in a source, and the agent finds '
                '60 of the remaining 80, then g = 20% and r = 75%. It succeeds on 60% of all questions.')
    st.subheader('One question. Four places to look.')
    cols = st.columns(4)
    for col, (source, name), description in zip(cols, SOURCES.items(), (
        'What the model learned during training. May be out of date.',
        'Retrieved pages. Can contain conflicting or incomplete evidence.',
        'Documents or structured records. Entity lookup and computation may be needed.',
        'Previous interactions. Past mistakes may be repeated.')):
        with col.container(border=True):
            st.markdown(f'### {source} · {name}')
            st.write(description)
    st.subheader('See the proposed source-first flow')
    st.markdown(decision_flow_html(source_first=True), unsafe_allow_html=True)
    st.caption('Source-first routing is an interactive policy prototype in Routing simulator. '
               'Experiments and Predict retain the existing four-source comparison pipeline.')
    st.subheader('Use the app, step by step')
    steps = [
        ('1. See why routing matters', 'Routing simulator',
         'Choose a scenario, adjust absence and retrieval skill, then step through the route and selected-source confidence.'),
        ('2. Prepare the evidence', 'Context studio',
         'Keep Sample selected. Press Compile context. Inspect the selected model, retained evidence, conflicts, and rejected models.'),
        ('3. Reproduce an experiment', 'Experiments',
         'Keep both samples. Select the independence approximation after reading it, then press Run experiment. Compare answer rate and accuracy.'),
        ('4. Understand uncertainty', 'Uncertainty lab',
         'Choose Controlled examples. Compare Each source is split with Sources disagree: the same overall uncertainty can have different causes.'),
        ('5. Inspect a prediction', 'Predict',
         'Choose Latest experiment or Bundled system. Keep sample queries and press Run predictions. Read the answer set and abstention reason.'),
    ]
    for i, (title, page, instruction) in enumerate(steps):
        with st.container(border=True):
            text, action = st.columns([4, 1])
            text.markdown(f'**{title}**')
            text.write(instruction)
            with action:
                _link(f'Open {page}', page, f'tour_{i}')
    with st.expander('Read a decision without the mathematics', expanded=True):
        st.markdown('**Answer:** one candidate remains after calibration.\n\n'
                    '**Abstain:** the system cannot narrow the answer down safely; inspect the evidence or add a source.\n\n'
                    '**Ø (outside):** the answer may be outside the supplied candidates. It does not prove the fact is absent from the world.\n\n'
                    '**Agreement:** repeated answers are consistent. That alone does not make them correct.')


def simulator():
    st.caption('POLICY PROTOTYPE · INTERACTIVE SIMULATION')
    st.title('Route first. Check confidence where you look.')
    st.write('Use query-side source estimates to choose one destination. Only then explore confidence '
             'for that destination. Change the controls to see when a better retriever helps and when a different source is needed.')
    st.info('Illustrative inputs and expected outcomes, not fitted KAP predictions or CRAG measurements. '
            'This policy runs here; it is not connected to the saved four-source prediction engine.')
    scenario = st.selectbox('Choose a question scenario', tuple(SCENARIOS), key='route_scenario')
    question, values = SCENARIOS[scenario]
    st.markdown(f'**Example question:** {question}')
    st.caption('K is the knowledge-base source label. KAP signals are query-side features used to estimate source suitability; '
               'they are available before answer generation. Here the sliders stand in for those estimates.')
    estimates = {}
    columns = st.columns(4)
    for col, (source, name), (g, r) in zip(columns, SOURCES.items(), values):
        with col.container(border=True):
            st.markdown(f'**{source} · {name}**')
            ghost = st.slider('Answer absent (g)', 0.0, 1.0, g, .01, key=f'{scenario}_{source}_g')
            reliability = st.slider('Finds it when present (r)', 0.0, 1.0, r, .01, key=f'{scenario}_{source}_r')
            estimates[source] = (ghost, reliability)
    penalty = st.slider('How much should retrieval cost matter?', 0.0, 1.0, .20, .05, key='route_penalty')
    costs = {'P': .10, 'W': .80, 'K': .35, 'M': .15}
    selected, rows = route_illustration(estimates, costs, penalty)
    frame = pd.DataFrame(rows)
    chart = alt.Chart(frame).mark_bar(cornerRadiusEnd=5).encode(
        y=alt.Y('Source:N', sort='-x', title=None),
        x=alt.X('Utility:Q', title='Expected useful retrieval minus weighted cost'),
        color=alt.condition(alt.datum.Source == (selected or ''), alt.value('#168a80'), alt.value('#bbcbd2')),
        tooltip=['Source', alt.Tooltip('Useful retrieval:Q', format='.1%'), 'Cost units', 'Utility'])
    st.altair_chart(chart.properties(height=190), use_container_width=True)
    st.caption('Teaching costs: P 0.10 · W 0.80 · K 0.35 · M 0.15. These are dimensionless assumptions, not provider prices. '
               'The highest positive utility wins; otherwise retrieval is skipped.')
    stage = st.slider('Walk through the decision', 1, 5, 1, key='route_stage')
    descriptions = [
        ('Understand the question', 'Estimate source coverage and retrieval difficulty using question-side metadata. No answer paths have been requested.'),
        ('Choose the destination', f'Selected {selected} · {SOURCES[selected]} using the utility ranking.' if selected else 'Every utility is nonpositive. Skip retrieval and abstain.'),
        ('Retrieve from that source', f'Only {selected} would receive retrieval and answer-path requests. Other source answer paths are skipped.' if selected else 'No source is selected, so there is no retrieval.'),
        ('Check selected-source confidence', 'Inspect repeated answers and the source absence estimate together. Agreement alone is insufficient.'),
        ('Answer or abstain', 'The teaching gate below combines retrieval potential with a manually supplied agreement score. Production uses independently calibrated prediction sets.'),
    ]
    title, description = descriptions[stage - 1]
    st.subheader(f'{stage:02d} · {title}')
    st.write(description)
    cols = st.columns(3)
    cols[0].metric('Selected source', selected or 'Abstain')
    cols[1].metric('Source destinations', '1 / 4' if selected else '0 / 4')
    cols[2].metric('Answer-path requests so far', '0 (illustration)')
    st.caption('Routing may still need cheap metadata probes. No live requests are made by this simulator; '
               'one source destination does not mean one model call.')
    if selected and stage >= 3:
        g, r = estimates[selected]
        st.subheader('What happens across 1,000 similar questions?')
        outcomes = pd.DataFrame([{'Outcome': name, 'Expected questions': mass * 1000}
                                 for name, mass in outcome_masses(g, r).items()])
        st.altair_chart(alt.Chart(outcomes).mark_bar(cornerRadiusEnd=5).encode(
            x=alt.X('Expected questions:Q', scale=alt.Scale(domain=[0, 1000])),
            y=alt.Y('Outcome:N', sort=None, title=None),
            color=alt.Color('Outcome:N', scale=alt.Scale(range=COLORS), legend=None),
            tooltip=['Outcome', alt.Tooltip('Expected questions:Q', format='.1f')]
        ).properties(height=150), use_container_width=True)
        st.caption('Analytic expectation: found = (1−g)r; present but missed = (1−g)(1−r); absent = g. '
                   'Holding g fixed while raising r moves questions from missed to found, never from absent to present.')
        if stage >= 4:
            agreement = st.slider(f'Illustrative answer agreement in {selected}', 0.0, 1.0, .80, .01, key='route_agreement')
            threshold = st.slider('Teaching gate threshold', 0.0, 1.0, .50, .01, key='route_threshold')
            score = (1 - g) * r * agreement
            st.metric('Teaching evidence score', f'{score:.1%}')
            st.caption('Score = (1−g) × r × agreement. This heuristic is not a calibrated probability of correctness. '
                       'A live integration would measure agreement only on the selected source.')
            if stage == 5:
                if score > threshold:
                    st.success('Teaching decision: allow an answer candidate for review. No actual answer has been generated.')
                else:
                    st.warning('Teaching decision: abstain. Add evidence or improve retrieval before answering.')
                st.download_button('Download this simulation', json.dumps({
                    'kind': 'illustrative_source_first_policy', 'question': question,
                    'selected_source': selected, 'ranking': rows, 'cost_penalty': penalty,
                    'agreement': agreement, 'teaching_score': score, 'threshold': threshold,
                    'decision': 'review_candidate' if score > threshold else 'abstain',
                    'live_calls': 0}, indent=2), file_name='routing-simulation.json', mime='application/json')
    st.subheader('The skill–coverage frontier')
    curve = pd.DataFrame([{'Reliability': i / 100, 'Useful retrieval': (1 - g) * i / 100,
                            'Absence assumption': f'g = {g:.0%}'}
                           for g in (0.1, 0.5, 0.9) for i in range(101)])
    st.altair_chart(alt.Chart(curve).mark_line(strokeWidth=3).encode(
        x=alt.X('Reliability:Q', axis=alt.Axis(format='.0%')),
        y=alt.Y('Useful retrieval:Q', scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format='.0%')),
        color=alt.Color('Absence assumption:N', scale=alt.Scale(range=COLORS)),
        tooltip=['Absence assumption', alt.Tooltip('Reliability:Q', format='.0%'),
                 alt.Tooltip('Useful retrieval:Q', format='.1%')]).properties(height=260), use_container_width=True)
    st.caption('Even perfect retrieval cannot recover information missing from the selected corpus. '
               'For computed answers, first check whether the source contains the records needed to derive the result.')


def algorithms():
    st.caption('ALGORITHM EXPLORER · FROM INTUITION TO IMPLEMENTATION')
    st.title('Open the black box.')
    st.write('Follow each component from its purpose to its mathematics. The labels below distinguish '
             'implemented library behavior, the routing prototype, and approaches described in the supplied report.')
    with st.expander('1 · Measure two different failure causes', expanded=True):
        st.write('With independently checked source-presence labels, count found answers, present-but-missed answers, '
                 'and absent answers. Fit source estimates using training data only.')
        st.latex(r'r_s = \frac{\mathrm{found}_s}{\mathrm{found}_s + \mathrm{missed}_s},\qquad g_s = \frac{\mathrm{absent}_s}{N_s}')
        st.warning('Unresolved entity links and answers requiring calculation need separate audits. '
                   'Neither a missing text match nor an unsuccessful lookup proves corpus absence. '
                   'If no corpus-present examples exist, reliability is undefined, not zero.')
    with st.expander('2 · Predict source estimates with KAP features'):
        st.write('Implemented: two logistic heads per source. Query-side features include scope fit, web metadata, '
                 'knowledge coverage, memory history, and estimated reasoning distance. Missing features receive '
                 'training-set imputations and explicit missing indicators.')
        st.latex(r'\hat r_s = \sigma(w_{r,s}^{\top}x+b_{r,s}),\quad \hat g_s = \sigma(w_{g,s}^{\top}x+b_{g,s}),\quad \sigma(z)=\frac{1}{1+e^{-z}}')
        st.code('Extract query-side features → impute from training data\n'
                'Predict g for each source\nPredict r conditional on answer presence\n'
                'Keep calibration and test labels out of these inputs', language=None)
        st.caption('Implementation: evidence/features.py, evidence/prediction.py, experiments/training.py. '
                   'The simulator substitutes manual estimates; it does not claim to train these heads.')
    with st.expander('3 · Route sources and route models'):
        st.write('Prototype source policy: rank expected useful retrieval after a cost penalty. '
                 'Generate paths only after selecting a source. This removes the need to sample every source merely to decide where to go.')
        st.latex(r's^*=\arg\max_s\left[(1-\hat g_s)\hat r_s-\lambda c_s\right]')
        st.write('Implemented model router: reject models failing capability, context, cost or latency limits. '
                 'Quality mode prioritizes declared suitability; economy mode prioritizes estimated cost. '
                 'Model size alone is not evidence of accuracy. The model cards supply the estimates.')
        st.code('Query → KAP estimates → select source → choose feasible model\n'
                '→ retrieve selected source → sample selected source → assess → answer / abstain', language=None)
        st.caption('Production integration still requires training and validating the source selector, '
                   'then adapting generation and calibration to the chosen-source policy. Existing fusion experiments sample P/W/K/M.')
    with st.expander('4 · Retrieve and rank passages'):
        st.write('The supplied report describes BM25 lexical shortlisting → embedding similarity ranking → '
                 'cross-encoder reranking → an optional language-model reader. These progressively narrow evidence '
                 'before generating an answer. This workspace replays recorded paths and does not implement that retrieval stack.')
        st.markdown('**BM25:** find passages sharing informative query terms.\n\n'
                    '**Embedding rank:** find passages with a similar meaning.\n\n'
                    '**Cross-encoder:** score the question and each passage together.\n\n'
                    '**Reader:** use the selected evidence to form an answer.')
    with st.expander('5 · Compare source fusion methods'):
        st.write('Implemented comparisons: Bayesian channel fusion, Dempster–Shafer evidence combination, '
                 'equal-source averaging, source majority, and parameters only. Sources keep separate g and r estimates.')
        st.latex(r'L_s(y)=(1-g_s)r_sp_s(y)+[1-(1-g_s)r_s]u(y)')
        st.latex(r'p(y\mid e)\propto p_0(y)\prod_s L_s(y)')
        st.write('The current reference Bayesian method uses a joint-absence prior. Its product-of-ghost-masses '
                 'approximation assumes independent corpus absence. Correlated sources can invalidate that approximation.')
        st.warning('The supplied PDF reports problems with a related fusion rule and proposes outside-mass and noisy-OR '
                   'alternatives. Those alternatives are not validated or substituted into this repository by this interface update. '
                   'Fusion remains a research comparison, not a demonstrated CRAG improvement.')
        st.caption('Noisy-OR, 1−∏[1−(1−gₛ)rₛ], describes at least one successful source under independence. '
                   'It is not by itself a distribution over candidate answers or a conflict-resolution rule.')
    with st.expander('6 · Calibrate, answer, or abstain'):
        st.write('Implemented: group-conditional split conformal calibration. Sort calibration scores '
                 '1−p(true answer), use a finite-sample quantile, then retain candidates below that threshold.')
        st.latex(r'k=\lceil(n+1)(1-\alpha)\rceil,\qquad \Gamma(x)=\{y:1-p(y\mid x)\leq q_k\}')
        st.write('Answer only when the set contains exactly one ordinary candidate. Abstain for an empty or ambiguous '
                 'set, Ø in the set, no feasible model, or total conflict. Sparse or unseen calibration groups retain the full frame.')
        st.caption('Coverage relies on exchangeability within groups and a fixed pipeline. '
                   'Changing routing requires fresh independent calibration. It does not guarantee each answer is correct.')


def research_snapshot():
    st.caption('RESEARCH NOTEBOOK · SUPPLIED EVIDENCE')
    st.title('What is measured, and what is still open?')
    st.info('These values are transcribed from the user-provided chat and PROJECT_FEEDBACK.pdf. '
            'They were not reproduced by this app. The sources describe different snapshots and must not be pooled.')
    snapshot = st.radio('Evidence snapshot', ('Chat · 18–19 Nov 2026', 'Supplied PDF · progress report'),
                        horizontal=True, key='research_snapshot')
    if snapshot.startswith('Chat'):
        st.write('Chat reports 2,706 web questions and 1,153 knowledge-base questions; no LLM available at that stage.')
        data = pd.DataFrame([{'Source': 'Web', 'Stage': 'Before abstention', 'CRAG score': -.70},
                             {'Source': 'Web', 'Stage': 'After abstention', 'CRAG score': .008},
                             {'Source': 'Knowledge base', 'Stage': 'Before abstention', 'CRAG score': -.55},
                             {'Source': 'Knowledge base', 'Stage': 'After abstention', 'CRAG score': .048}])
        st.altair_chart(alt.Chart(data).mark_bar().encode(
            x='Source:N', xOffset='Stage:N', y=alt.Y('CRAG score:Q', scale=alt.Scale(domain=[-1, 1])),
            color=alt.Color('Stage:N', scale=alt.Scale(range=['#7184b1', '#168a80'])),
            tooltip=['Source', 'Stage', 'CRAG score']).properties(height=290), use_container_width=True)
        st.caption('Chat-reported scores. Positive values are small; answer rates and uncertainty intervals are not supplied. '
                   'This graph does not establish broad answer usefulness.')
        st.write('The chat also reports combined-source ghost mass of 0.28 versus 0.35 and 0.65 individually. '
                 'That is a reported coverage comparison; the chat explicitly says answer fusion does not yet work.')
    else:
        st.write('PDF §2 reports 2,706 questions for each of W and K and a local 1.5B reader. '
                 'This differs from the chat snapshot. No run identifiers are supplied to reconcile them.')
        data = pd.DataFrame({'Agent': ['Keyword', 'Semantic retrieval', 'LLM reader'],
                             'Reliability': [.066, .222, .286], 'Ghost mass': [.318, .326, .343]})
        long = data.melt('Agent', var_name='Quantity', value_name='Rate')
        st.altair_chart(alt.Chart(long).mark_line(point=True, strokeWidth=3).encode(
            x=alt.X('Agent:N', sort=list(data.Agent)),
            y=alt.Y('Rate:Q', axis=alt.Axis(format='.0%'), scale=alt.Scale(domain=[0, .4])),
            color=alt.Color('Quantity:N', scale=alt.Scale(range=['#d69a42', '#168a80'])),
            tooltip=['Agent', 'Quantity', alt.Tooltip('Rate:Q', format='.1%')]
        ).properties(height=290), use_container_width=True)
        st.caption('PDF §3.1 values. This is an agent-change comparison, not proof of statistical independence. '
                   'The report says the fixed-agent corpus-expansion experiment remains in progress.')
        st.write('PDF §4.2 reports a weaker sampling signal for presence (AUROC 0.639 on 300 questions), '
                 'so the strong claim that sampling cannot detect absence should not be presented as established.')
    st.subheader('Questions the system should surface')
    st.dataframe(pd.DataFrame([
        {'Issue': 'Entity linking', 'Why it matters': 'Failed lookup can be mistaken for missing knowledge.', 'Next evidence': 'Audit unresolved entities against graph membership.'},
        {'Issue': 'Computed answers', 'Why it matters': 'Records may exist even when the final answer text does not.', 'Next evidence': 'Evaluate lookup and aggregation tasks separately.'},
        {'Issue': 'Web coverage probe', 'Why it matters': 'Fixed page counts can remove useful predictive variation.', 'Next evidence': 'Test genuinely varying query-side metadata.'},
        {'Issue': 'Routing and fusion', 'Why it matters': 'Better coverage does not ensure correct combined answers.', 'Next evidence': 'Evaluate source-first routing and fusion on independent held-out data.'},
    ]), hide_index=True, width='stretch')
    st.caption('Novelty is a research hypothesis. The supplied material alone cannot establish that no prior paper studies these ideas.')
