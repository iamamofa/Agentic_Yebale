"""Responsive, accessible process diagram for the overview workspace."""

from html import escape


STAGES = (
    ('01', 'Prepare', 'Question, instructions, and source context.',
     '<rect x="5" y="4" width="14" height="17" rx="2"/><path d="M9 9h6M9 13h6M9 17h3"/>'),
    ('02', 'Route', 'Choose a model within capability and budget limits.',
     '<path d="M5 12h6m0 0 5-6m-5 6 5 6M13 6h4v4m-4 8h4v-4"/>'),
    ('03', 'Generate paths', 'Explore multiple answers from P, W, K, and M.',
     '<circle cx="6" cy="12" r="2"/><circle cx="18" cy="5" r="2"/><circle cx="18" cy="12" r="2"/><circle cx="18" cy="19" r="2"/><path d="m8 11 8-5m-8 6h8m-8 1 8 5"/>'),
    ('04', 'Fuse evidence', 'Combine source estimates and inspect conflicts.',
     '<path d="m4 5 7 7-7 7m7-7h9M4 12h7m5-4 4 4-4 4"/>'),
    ('05', 'Calibrate & decide', 'Build prediction sets and make a final decision.',
     '<path d="m12 3 9 9-9 9-9-9 9-9Z"/><path d="m8.5 12 2.2 2.2 4.8-4.8"/>'),
)


def decision_flow_html(source_first=False):
    stages = STAGES
    if source_first:
        descriptions = (
            ('Read the question', 'Estimate source coverage and retrieval skill from query-side signals.'),
            ('Select a source', 'Use KAP estimates to choose where to look before generating answers.'),
            ('Retrieve & sample', 'Consult the selected source and inspect its answer paths.'),
            ('Assess evidence', 'Check absence risk, selected-source agreement, and conflicts.'),
            ('Answer or abstain', 'Return a supported candidate or explain why more evidence is needed.'),
        )
        stages = tuple((stage[0], title, description, stage[3])
                       for stage, (title, description) in zip(STAGES, descriptions))
    cards = []
    for number, title, description, icon in stages:
        outcomes = ('<div class="flow-outcomes"><span>Answer</span><span>Abstain</span></div>'
                    if number == '05' else '')
        cards.append(
            f'<li class="flow-stage{" flow-stage-final" if number == "05" else ""}">'
            f'<div class="flow-card-top"><span class="flow-icon" aria-hidden="true">'
            f'{number}</span>'
            f'<span class="flow-step">STEP</span></div>'
            f'<h4>{escape(title)}</h4><p>{escape(description)}</p>{outcomes}</li>'
        )
    return ('<ol class="decision-flow" aria-label="Five stages from context to a decision">'
            + ''.join(cards) + '</ol>')
