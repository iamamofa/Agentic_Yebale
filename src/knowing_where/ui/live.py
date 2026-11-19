"""User-triggered Hugging Face inference with session-only credentials."""

import hashlib
import json
import os

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from ..providers.huggingface import HuggingFaceClient, HuggingFaceError, run_live_query

DEFAULT_MODEL = 'Qwen/Qwen3.5-9B'
SOURCE_NAMES = {'P': 'P · Model knowledge', 'W': 'W · Supplied web passages',
                'K': 'K · Supplied knowledge-base evidence', 'M': 'M · Supplied memory'}


def configured_token():
    token = os.environ.get('HF_TOKEN', '')
    if token:
        return token
    try:
        return str(st.secrets.get('HF_TOKEN', ''))
    except (StreamlitSecretNotFoundError, FileNotFoundError):
        return ''


def _forget():
    st.session_state.pop('hf_token_input', None)
    st.session_state.pop('hf_live_result', None)
    st.session_state.pop('hf_live_signature', None)


def _signature(model, question, source, evidence, paths, max_tokens, temperature):
    return hashlib.sha256(json.dumps([model, question, source, evidence, paths, max_tokens,
                                     temperature]).encode()).hexdigest()


def live_models():
    st.caption('LIVE INFERENCE · HUGGING FACE')
    st.title('Ask a real model. Inspect its evidence.')
    st.write('Connect to a hosted chat model, choose one source, and inspect the actual response. '
             'Only the source you select is sent to the reader.')
    with st.expander('1 · Connect Hugging Face', expanded=True):
        st.markdown('Create a token at [Hugging Face settings](https://huggingface.co/settings/tokens) '
                    'with **Inference Providers** permission. Enter it here or configure `HF_TOKEN` '
                    'in the server environment or Streamlit secrets.')
        token_input = st.text_input('Hugging Face token', type='password', key='hf_token_input',
                                   help='Kept in this Streamlit session. Not written to files or included in downloads.')
        server_token = configured_token()
        token = token_input.strip() or server_token
        if token:
            st.caption('Credential available from this session.' if token_input.strip()
                       else 'Credential available from server configuration.')
        else:
            st.info('Add a token to enable live generation. Model catalogue browsing works without one.')
        st.button('Forget session token and live results', on_click=_forget, key='hf_forget')
        st.caption('Forgetting the session token does not remove a token configured on the server.')
        if st.button('Load available models', key='hf_catalogue'):
            try:
                with st.spinner('Loading the public Hugging Face model catalogue…'):
                    st.session_state['hf_available_models'] = HuggingFaceClient().list_models()
                st.success('Available model catalogue loaded.')
            except HuggingFaceError as error:
                st.error(str(error))
        choices = st.session_state.get('hf_available_models', [DEFAULT_MODEL])
        choices = [*choices, 'Enter a model ID']
        selected = st.selectbox('Hosted chat model', choices,
                               index=choices.index(DEFAULT_MODEL) if DEFAULT_MODEL in choices else 0,
                               key='hf_model_choice')
        model = st.text_input('Custom model ID', key='hf_custom_model') if selected == 'Enter a model ID' else selected
        st.caption('Only models currently served by Hugging Face Inference Providers can run through this connection. '
                   'Availability, gated access, and charges depend on the model, provider, and your account.')
        if st.button('Test model connection', key='hf_test'):
            try:
                with st.spinner('Sending one short test request…'):
                    result = HuggingFaceClient(token).chat(model, [
                        {'role': 'user', 'content': 'Reply with the single word connected.'}
                    ], max_tokens=32, temperature=0)
                st.success(f'Live response received from {model}.')
                st.text(result['text'])
            except HuggingFaceError as error:
                st.error(str(error))
    st.subheader('2 · Choose your question and source')
    source = st.selectbox('Use this source only', tuple(SOURCE_NAMES), index=2,
                          format_func=SOURCE_NAMES.get, key='hf_source')
    question = st.text_area('Your question', value='What is the Atlas travel reimbursement limit?',
                            max_chars=5000, key='hf_question')
    evidence = ''
    if source != 'P':
        seed_evidence = 'Atlas travel policy: The reimbursement limit is 2,500 MUR per approved trip.'
        if st.session_state.get('context_result'):
            if st.checkbox('Use evidence from my latest compiled context', key='hf_use_context'):
                report = st.session_state['context_result']
                packet = next((candidate['context'] for candidate in report['candidates']
                               if candidate['model_id'] == report['selected_model_id']), None)
                if packet:
                    origins = {doc['id']: doc['source'] for doc in packet['documents']}
                    seed_evidence = '\n\n'.join(excerpt['text'] for excerpt in packet['excerpts']
                                                if origins[excerpt['document_id']] == source)
                    if not seed_evidence:
                        st.warning('This compiled packet has no evidence for the selected source.')
                else:
                    seed_evidence = ''
        evidence = st.text_area('Evidence for the selected source', value=seed_evidence, height=160,
                                max_chars=40000, key=f'hf_evidence_{source}_{hashlib.sha256(seed_evidence.encode()).hexdigest()[:12]}')
        st.caption('Paste the passages or records the reader should use. W does not perform a web search; '
                   'K does not automatically connect a database; M does not read other chats. '
                   'The default Atlas passage is fictional sample evidence.')
    else:
        st.caption('P uses the model’s learned knowledge. Evidence from your other source fields is not sent.')
    st.subheader('3 · Generate and inspect')
    cols = st.columns(3)
    paths = cols[0].number_input('Answer paths (API calls)', min_value=1, max_value=5, value=1,
                                key='hf_paths', step=1)
    max_tokens = cols[1].number_input('Maximum output tokens per path', min_value=32, max_value=2048,
                                     value=256, step=32, key='hf_max_tokens')
    temperature = cols[2].slider('Temperature', 0.0, 1.0, .2, .1, key='hf_temperature')
    st.caption(f'Run sends {paths} generation request(s) to Hugging Face and its inference provider. '
               'Your question and selected evidence leave this machine. Charges or credits may apply. '
               'There are no automatic retries; the token limit applies to each request.')
    signature = _signature(model, question, source, evidence, paths, max_tokens, temperature)
    if st.button('Generate live answer', type='primary', key='hf_generate'):
        st.session_state.pop('hf_live_result', None)
        st.session_state.pop('hf_live_signature', None)
        try:
            with st.spinner(f'Running {paths} live answer path(s) on source {source}…'):
                result = run_live_query(HuggingFaceClient(token), model=model, question=question,
                                        source=source, evidence=evidence, paths=int(paths),
                                        max_tokens=int(max_tokens), temperature=temperature)
            st.session_state['hf_live_result'] = result
            st.session_state['hf_live_signature'] = signature
        except HuggingFaceError as error:
            st.error(str(error))
    result = st.session_state.get('hf_live_result')
    if not result:
        return
    st.divider()
    if signature != st.session_state.get('hf_live_signature'):
        st.info('Inputs changed. Showing the last completed live run; generate again to apply your changes.')
    st.caption(f"Last completed run · {result['model']} · source {result['source']} · {result['paths']} path(s)")
    if result['model_abstained']:
        st.warning('The model declined to answer from the available information.')
    else:
        st.subheader('Model response')
        st.text(result['answer'])
    if any(sample.get('finish_reason') == 'length' for sample in result['samples']):
        st.warning('At least one response reached the output limit and may be incomplete.')
    if result['exact_text_agreement'] is not None:
        st.metric('Exact-text agreement', f"{result['exact_text_agreement']:.0%}")
        st.caption('Whitespace and letter case are normalized. Different phrasings may count as different answers. '
                   'Agreement does not measure correctness or evidence support.')
    else:
        st.caption('One sample: agreement is not estimated. Increase answer paths to compare repeated responses.')
    st.info('Live reader output is uncalibrated. It does not estimate ghost mass or reliability, verify quoted evidence, '
            'or reuse the synthetic demo’s conformal guarantees. The saved research pipeline remains separate.')
    with st.expander('Inspect every answer path and token usage'):
        for i, sample in enumerate(result['samples'], 1):
            st.markdown(f'**Path {i}**')
            st.text(sample['text'])
        st.dataframe(pd.DataFrame([{'Path': i, 'Model': sample['model'],
                                   'Finish reason': sample['finish_reason'], **sample['usage']}
                                  for i, sample in enumerate(result['samples'], 1)]),
                     hide_index=True, width='stretch')
    st.download_button('Download live results', json.dumps(result, indent=2),
                       file_name='huggingface-live-results.json', mime='application/json', key='hf_download')
