"""Guided learning, sourced reading notes, and a reproducible study planner."""

import csv
import io

import streamlit as st

from .learning import _link

# Curated foundations, rather than a claim of an exhaustive literature search.
PAPERS = (
    dict(theme='Retrieval', author='Lewis et al. · 2020',
         title='Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks',
         url='https://arxiv.org/abs/2005.11401',
         idea='Combine knowledge stored in model parameters with retrieved external passages.',
         connection='Compare P (model knowledge) with W and K (external evidence) in this workbench.',
         limit='Retrieving a passage does not establish that it contains the answer.',
         question='Did retrieval miss the evidence, or was the evidence absent?', page='Context studio'),
    dict(theme='Agreement', author='Wang et al. · 2022 preprint / ICLR 2023',
         title='Self-Consistency Improves Chain of Thought Reasoning in Language Models',
         url='https://arxiv.org/abs/2203.11171',
         idea='Sample several reasoning paths and aggregate their answers.',
         connection='The experiment engine inspects distributions of canonical answer paths; the live reader reports exact-text agreement.',
         limit='Repeated answers can share the same error. Agreement alone does not prove correctness.',
         question='Can a source repeatedly give the same answer while lacking supporting evidence?', page='Uncertainty lab'),
    dict(theme='Calibration', author='Angelopoulos & Bates · 2021',
         title='A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification',
         url='https://arxiv.org/abs/2107.07511',
         idea='Use held-out calibration data to construct prediction sets with a target coverage level under suitable assumptions.',
         connection='The saved pipeline fits group-specific (Mondrian) thresholds on the calibration partition.',
         limit='Coverage depends on exchangeability and a frozen pipeline; it is not a correctness guarantee for an individual answer.',
         question='Does useful coverage remain when we report actual answers separately from the outside option Ø?', page='Predict'),
    dict(theme='Evaluation', author='Yang et al. · 2024',
         title='CRAG — Comprehensive RAG Benchmark',
         url='https://arxiv.org/abs/2406.04744',
         idea='Evaluate factual question answering across diverse domains, question types, and changing information needs.',
         connection='The Research notebook records supplied CRAG-related progress; bundled experiments use synthetic fixtures.',
         limit='Local replay outcomes do not establish performance on CRAG or on live models.',
         question='Which failure types persist across domains, sources, and retrieval agents?', page='Research notebook'),
)

JOURNEY = (
    ('Meet the four sources', 'Start here',
     'Read “The idea”. Explain P, W, K, and M in your own words.',
     'You can tell model knowledge, web evidence, a knowledge base, and memory apart.'),
    ('Make a routing decision', 'Routing simulator',
     'Choose “A fact in company documents” and move the walkthrough to step 5. Change the teaching threshold.',
     'You can explain why the same source can lead to an answer candidate or abstention.'),
    ('Inspect your first evidence packet', 'Context studio',
     'Keep Sample selected, press Compile context, then inspect Claim conflicts and Exact payload.',
     'You can find the retained evidence, a declared conflict, and the exact model input.'),
    ('Run a comparison', 'Experiments',
     'Keep the samples, select the independence approximation, then press Run experiment. Read accuracy alongside answer rate.',
     'You can distinguish the training, calibration, and test partitions.'),
    ('Read a prediction', 'Predict',
     'Choose Latest experiment (or Bundled system), keep the sample queries, then press Run predictions.',
     'You can explain the prediction set, Ø, and the reason for answering or abstaining.'),
)

METHOD = (
    ('Define the question', 'Write a falsifiable claim',
     'Example: holding the source corpus fixed, a stronger retriever improves success when evidence is present.',
     'Record the source snapshot, question families, reference answers, and separately audited presence labels. A failed search is not an absence label.',
     'A question list and annotation protocol.', 'Start here'),
    ('Separate the data', 'Split before fitting',
     'Keep related questions in the same split_group. The bundled example uses group-isolated 40% train / 20% calibration / 40% test partitions.',
     'Fit source heads only on training examples. Generation receives no gold answers, presence labels, or split-family labels.',
     'A seeded split manifest with no family overlap.', 'Experiments'),
    ('Prepare and sample', 'Keep the evidence trail',
     'Compile budgeted context, retain provenance, and replay canonical answer paths for P, W, K, and M. The saved engine compares all four sources.',
     'The source-first simulator is a teaching prototype. W/K/M in Live models read supplied evidence; they do not automatically search external services.',
     'Context audit, model profile, path count, and generation budget.', 'Context studio'),
    ('Fit and combine', 'Compare the same held-out cases',
     'Learn source reliability and absence estimates on training data, then compare Bayesian fusion, Dempster–Shafer, equal sources, majority, and parametric-only scoring.',
     'Record the corpus-absence independence approximation. Correlated sources and answer agreement can produce overconfidence.',
     'Frozen coefficients and baseline configurations.', 'Algorithm explorer'),
    ('Calibrate and decide', 'Allow uncertainty to stay visible',
     'Use only the calibration partition for group-specific thresholds. A singleton canonical answer can be returned; Ø, ambiguity, or total conflict leads to abstention.',
     'Changing the model, prompt, retrieval policy, or answer mapping requires fresh validation and calibration.',
     'Calibration thresholds, group counts, alpha, and fitted artifact.', 'Predict'),
    ('Evaluate and report', 'Read usefulness and coverage together',
     'Evaluate once on the held-out test partition. Report point accuracy, answer rate, selective accuracy, actual answer coverage, mapped coverage, and mean set size.',
     'Mapped coverage can include Ø. Report sparse groups, entity lookup failures, source dependence, and the limits of synthetic replay.',
     'Report JSON, per-group results, limitations, and reproducibility notes.', 'Uncertainty lab'),
)

GLOSSARY = {
    'Ghost mass (g)': 'Estimated chance that a source does not contain the answer. Establishing absence requires an independent audit.',
    'Reliability (r)': 'Chance of finding the answer conditional on it being present. Useful retrieval in the teaching model is (1−g)r.',
    'KAP signals': 'Question-side features used by the source heads. They are available before answer generation; K alone is the knowledge-base label.',
    'Prediction set': 'The candidates retained after applying calibrated thresholds. More than one candidate means unresolved ambiguity.',
    'Outside option (Ø)': 'The reference answer may be outside the supplied candidate set. Including Ø leads the saved policy to abstain.',
    'Selective accuracy': 'Accuracy among answered cases. Always read it alongside answer rate so extensive abstention stays visible.',
    'Actual vs mapped coverage': 'Actual coverage checks the canonical reference answer. Mapped coverage counts Ø when that answer is absent from the candidates.',
    'Replay vs live': 'Replay uses recorded paths. Live models calls a hosted reader using your provider account and has no attached research calibration.',
}


def guide_hub():
    st.caption('KNOWING WHERE / FIELD GUIDE')
    st.title('From first visit to a defensible study.')
    st.write('Learn the language, connect the literature, and follow a method you can explain. '
             'Take one small step, then try it in the workbench.')
    with st.container(horizontal=True):
        st.badge('Learn by doing', icon=':material/explore:', color='green')
        st.badge('4 foundational readings', icon=':material/local_library:', color='blue')
        st.badge('6 research stages', icon=':material/route:', color='orange')
    section = st.segmented_control('Choose your guide',
        ['First visit', 'Literature review', 'Methodology', 'Study planner'],
        default='First visit', key='guide_section', persist_state='session')
    st.divider()
    if section == 'Literature review':
        _literature()
    elif section == 'Methodology':
        _methodology()
    elif section == 'Study planner':
        _planner()
    else:
        _first_visit()
    with st.expander('Pocket glossary · translate the research language'):
        term = st.selectbox('Look up a term', tuple(GLOSSARY), key='guide_term', persist_state='session')
        st.write(GLOSSARY[term or next(iter(GLOSSARY))])
    st.caption('Guides and checklist progress stay in this session. Download your study plan to keep a copy.')


def _first_visit():
    st.subheader('Your first evidence journey')
    st.write('Use the bundled samples for these five stops. Mark a step when you can explain its takeaway; '
             'the checklist records your own progress.')
    progress = st.empty()
    completed = 0
    for index, (title, page, instruction, takeaway) in enumerate(JOURNEY):
        with st.container(border=True):
            body, action = st.columns([3, 1], gap='medium')
            with body:
                st.caption(f'STOP {index + 1:02d} / 05')
                st.subheader(title, anchor=False)
                st.write(instruction)
                st.caption(f'You are ready when: {takeaway}')
                completed += int(st.checkbox('I can explain this step', key=f'guide_done_{index}', persist_state='session'))
            with action:
                _link(f'Open {page}', page, f'journey_open_{index}')
    progress.progress(completed / len(JOURNEY), text=f'{completed} of {len(JOURNEY)} stops completed')
    if completed == len(JOURNEY):
        st.success('You have completed the starter journey. Continue with Literature review, then Methodology to plan your own study.')
    st.info('Optional next step: Live models uses a Hugging Face account. Start there only when you want to send a question and supplied evidence to a hosted model.')
    with st.container(border=True):
        st.markdown('#### :material/quiz: Check your intuition')
        answer = st.segmented_control('A model gives the same answer ten times. What have you established?',
            ['Agreement', 'Correctness', 'Source presence'], key='guide_quiz', persist_state='session')
        if answer == 'Agreement':
            st.success('Exactly. Repetition establishes agreement; independent evidence is still needed to assess correctness and source presence.')
        elif answer:
            st.info('Try again: ten matching answers show agreement. They can still repeat a shared error or an unsupported guess.')


def _literature():
    st.subheader('A reading map, with a purpose')
    st.write('Follow four foundations: retrieve evidence → compare answers → calibrate uncertainty → evaluate realistic questions. '
             'These curated reading notes are a starting point for a review, not an exhaustive survey.')
    theme = st.pills('Reading lens', ['All', 'Retrieval', 'Agreement', 'Calibration', 'Evaluation'],
                     default='All', key='reading_lens', persist_state='session') or 'All'
    for index, paper in enumerate(PAPERS, 1):
        if theme != 'All' and theme != paper['theme']:
            continue
        with st.container(border=True):
            st.caption(f"{index:02d} / {paper['theme'].upper()} · {paper['author']}")
            st.subheader(paper['title'], anchor=False)
            left, right = st.columns([3, 2], gap='medium')
            with left:
                st.markdown(f"**Core idea**\n\n{paper['idea']}")
                st.markdown(f"**Connection to this app**\n\n{paper['connection']}")
                st.link_button('Read the original paper', paper['url'], icon=':material/open_in_new:')
            with right:
                st.markdown(f"**Read critically**\n\n{paper['limit']}")
                st.caption(f"Review question: {paper['question']}")
                _link(f"Explore {paper['theme'].lower()}", paper['page'], f'reading_open_{index}')
    st.subheader('Turn reading into a literature review')
    for title, text in (
        ('01 · Set the scope', 'Define the question, databases, search strings, date range, and inclusion/exclusion criteria. Log the search date and screening decisions.'),
        ('02 · Build an evidence matrix', 'For each paper record its task, dataset, source assumptions, baseline, metrics, findings, and limitations. Separate author claims from your interpretation.'),
        ('03 · Synthesize across themes', 'Compare how studies treat retrieval failure, source absence, repeated answers, and abstention. Explain disagreements rather than listing abstracts.'),
        ('04 · State a testable gap', 'Ask whether source coverage and conditional retrieval skill can be measured separately. Treat this as this project’s research question; a broader review must establish any novelty claim.'),
    ):
        st.markdown(f'**{title}** — {text}')
    matrix = io.StringIO()
    writer = csv.writer(matrix)
    writer.writerow(['Paper', 'Authors / year', 'Task', 'Dataset', 'Source assumptions',
                     'Baseline', 'Metrics', 'Finding', 'Limitation', 'Connection', 'URL'])
    for paper in PAPERS:
        writer.writerow([paper['title'], paper['author'], '', '', '', '', '', '', '', '', paper['url']])
    st.download_button('Download a literature matrix', matrix.getvalue(), file_name='literature-matrix.csv',
                       mime='text/csv', icon=':material/download:')


def _methodology():
    st.subheader('One method, six checkpoints')
    st.write('Follow the saved experiment pipeline from a question to a held-out result. '
             'Choose a stage to see its purpose, the evidence to keep, and the workspace to try.')
    stage = st.select_slider('Research stage', options=list(range(1, 7)), value=1,
                             format_func=lambda n: f'{n:02d} · {METHOD[n - 1][0]}',
                             key='method_stage', persist_state='session')
    title, subtitle, action, caution, output, page = METHOD[stage - 1]
    st.progress(stage / len(METHOD), text=f'Stage {stage} of {len(METHOD)} · {title}')
    with st.container(border=True):
        st.caption(f'CHECKPOINT {stage:02d}')
        st.subheader(subtitle)
        st.write(action)
        left, right = st.columns(2, gap='medium')
        with left:
            st.markdown('**Keep as evidence**')
            st.write(output)
        with right:
            st.markdown('**Check before moving on**')
            st.write(caution)
        _link(f'Try this in {page}', page, 'method_open')
    st.subheader('Three partitions, three different jobs')
    for col, label, amount, detail in zip(st.columns(3),
            ['Train', 'Calibrate', 'Test'], ['40%', '20%', '40%'],
            ['Learn source heads and freeze coefficients.', 'Set thresholds using held-out calibration cases.', 'Measure outcomes after the pipeline is frozen.']):
        with col.container(border=True, height='stretch'):
            st.metric(label, amount)
            st.caption(detail)
    st.caption('These ratios describe the bundled synthetic example. Split by question family, not by individual paraphrase.')
    st.info('Worked teaching example: if 30 of 100 questions have no answer in a source and the retriever succeeds on 80% of the remaining 70, expect 56 found, 14 missed, and 30 absent. More retrieval skill can recover the 14 misses; the 30 absent answers need another source.')


def _planner():
    st.subheader('Leave with a plan you can use')
    st.write('Draft a compact study brief and export it with the method checkpoints and reading list. '
             'Your notes stay in your session until you download them.')
    with st.form('study_brief_form'):
        question = st.text_area('Your research question',
            value='Does improving retrieval skill reduce missed answers while source absence stays fixed?',
            key='plan_question', persist_state='session')
        sources = st.multiselect('Sources to study', ['P · Model knowledge', 'W · Web', 'K · Knowledge base', 'M · Memory'],
                                 default=['W · Web', 'K · Knowledge base'], key='plan_sources', persist_state='session')
        notes = st.text_area('Dataset, comparison, and limitations',
            placeholder='Name the corpus snapshot, presence-label audit, retriever variants, family split, and metrics you will report.',
            key='plan_notes', persist_state='session')
        prepared = st.form_submit_button('Prepare my study brief', type='primary')
    content = ['# Knowing Where · Study brief', '', '## Research question', question,
               '', '## Sources', ', '.join(sources) or 'To be specified',
               '', '## Dataset and study notes', notes or 'To be specified',
               '', '## Method checkpoints']
    content += [f'\n### {i}. {stage[0]}\n{stage[2]}\n\nKeep: {stage[4]}\n\nCheck: {stage[3]}'
                for i, stage in enumerate(METHOD, 1)]
    content += ['', '## Reading list'] + [f'- {p["author"]}. [{p["title"]}]({p["url"]}).' for p in PAPERS]
    content += ['', '## Reporting reminder',
                'Report accuracy, answer rate, selective accuracy, actual and mapped coverage, and mean set size together. '
                'Distinguish synthetic replay, supplied research reports, the routing prototype, and live reader outputs.']
    if prepared:
        if question.strip():
            st.session_state['guide_prepared_brief'] = '\n'.join(content)
            st.success('Your study brief is ready to download.')
        else:
            st.session_state.pop('guide_prepared_brief', None)
            st.info('Add a research question, then prepare your brief.')
    brief = st.session_state.get('guide_prepared_brief')
    if brief:
        st.download_button('Download my study brief', brief, file_name='knowing-where-study-brief.md',
                           mime='text/markdown', icon=':material/download:')
        st.caption('This downloads your last prepared brief. After editing, select Prepare my study brief again.')
    st.caption('Export is a local download. It does not start an experiment or call a model.')
