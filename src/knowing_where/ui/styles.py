"""Shared visual system for the evidence workbench."""

STYLE = '''<style>
:root { --ink:#132d3e; --teal:#168a80; --gold:#d69a42; }
.stApp {background:radial-gradient(ellipse at 85% 0%,#e8f5ef 0%,transparent 42%),#f6f8fb;}
.block-container {max-width:1320px;padding-top:2.6rem;padding-bottom:3rem;}
h1,h2,h3 {font-family:'Avenir Next','Segoe UI',sans-serif;color:var(--ink);letter-spacing:-.035em;}
h1 {font-size:2.9rem!important;line-height:1.12!important;font-weight:800!important;max-width:900px;}
p,label,button {font-family:'Avenir Next','Segoe UI',sans-serif;}
[data-testid="stCaptionContainer"] {color:#697c8e;}
[data-testid="stSidebar"] {background:linear-gradient(165deg,#102f40 0%,#153d4c 60%,#0c2738 100%);border-right:0;}
[data-testid="stSidebar"] h3,[data-testid="stSidebar"] p,[data-testid="stSidebar"] label {color:#eef8f7!important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {color:#a9c6d0!important;font-size:.77rem;letter-spacing:.03em;}
[data-testid="stSidebar"] hr {border-color:#42616c;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {padding:.5rem .7rem;border-radius:9px;margin-bottom:3px;}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {background:#244e5d;}
[data-testid="stSidebar"] button {background:#244e5d;border:1px solid #51707b;color:#fff;}
[data-testid="stMetric"] {background:#fff;border:1px solid #e4eaef;border-radius:15px;padding:18px 20px;box-shadow:0 3px 12px #16384c04;}
[data-testid="stMetricValue"] {font-family:'Avenir Next','Segoe UI',sans-serif;font-weight:750;color:#163e4c;}
[data-testid="stForm"] {background:#fff;border:1px solid #e2e9ed;border-radius:16px;padding:22px;}
[data-testid="stExpander"] {background:#ffffffab;border:1px solid #e2e9ed;border-radius:12px;}
[data-testid="stAlert"] {border-radius:12px;}
[data-testid="stTabs"] [role="tablist"] {gap:18px;border-bottom:1px solid #dfe8ed;}
[data-testid="stTabs"] [role="tab"] {font-weight:600;}
div.stButton>button[kind="primary"] {border-radius:9px;background:#137a70;border:0;padding:.65rem 1.1rem;box-shadow:0 4px 10px #137a7018;}
div.stDownloadButton>button {border-radius:9px;}
.uncertainty-orbit {width:188px;height:188px;border-radius:50%;margin:12px auto 18px;padding:13px;
  background:conic-gradient(from -90deg,#168a80 0deg,#168a80 var(--orbit-angle),#e3ecee var(--orbit-angle),#e3ecee 360deg);
  box-shadow:0 0 0 9px #fff,0 8px 32px #1437480c;}
.orbit-center {width:100%;height:100%;border-radius:50%;background:#fff;display:flex;flex-direction:column;align-items:center;justify-content:center;}
.orbit-number {font-family:'Avenir Next','Segoe UI',sans-serif;font-size:3.25rem;font-weight:800;line-height:1.05;color:#153e4c;letter-spacing:-.05em;}
.orbit-number small {font-size:1.2rem;margin-left:3px;color:#78949a;}
.orbit-label {font-size:.6rem;font-weight:700;letter-spacing:.12em;color:#70888f;margin-top:9px;}
.decision-strip {display:flex;align-items:center;gap:20px;background:#fff;border:1px solid #e0e8ed;
  border-left:4px solid #168a80;border-radius:10px;padding:13px 18px;margin:10px 0 24px;color:#617888;font-size:.85rem;}
.decision-strip strong {flex:1;color:#153e4c;font-size:.95rem;}
.source-chips {display:flex;gap:7px;flex-wrap:wrap;margin:12px 0 22px;}
.source-chips span {border:1px solid #476773;background:#234a58;color:#ddf1ee;border-radius:20px;padding:4px 10px;font-size:.72rem;}
@media(max-width:760px){h1{font-size:2.1rem!important}.block-container{padding-top:1.5rem}.decision-strip{flex-wrap:wrap;gap:8px}.uncertainty-orbit{width:160px;height:160px}}
.decision-flow {list-style:none!important;margin:8px 0 28px!important;padding:0!important;
  display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:26px;}
.decision-flow .flow-stage {position:relative;margin:0!important;padding:19px 16px!important;
  min-height:212px;background:#fff;border:1px solid #dce8e9;border-radius:15px;
  box-shadow:0 4px 16px #133c4c04;}
.flow-stage:not(:last-child)::after {content:"";position:absolute;right:-22px;top:48%;
  width:16px;height:8px;background:#75a9a5;clip-path:polygon(0 35%,65% 35%,65% 0,100% 50%,65% 100%,65% 65%,0 65%);}
.flow-card-top {display:flex;align-items:center;justify-content:space-between;margin-bottom:16px;}
.flow-icon {display:flex;align-items:center;justify-content:center;width:36px;height:36px;
  border-radius:10px;background:#e8f5f1;color:#168a80;}
.flow-icon svg {width:23px;height:23px;}
.flow-step {font-size:.72rem;letter-spacing:.08em;font-weight:700;color:#7e969e;}
.decision-flow h4 {font-size:.96rem;font-weight:750;line-height:1.35;color:#163e4c;margin:0 0 8px!important;}
.decision-flow p {font-size:.76rem!important;line-height:1.6;color:#647b88;margin:0!important;}
.decision-flow .flow-stage-final {background:linear-gradient(145deg,#123d4c,#155f61);border-color:#155f61;}
.flow-stage-final h4,.flow-stage-final p {color:#eef9f6;}
.flow-stage-final .flow-step {color:#afcfc9;}
.flow-stage-final .flow-icon {background:#ffffff18;color:#b9ebe0;}
.flow-outcomes {display:flex;gap:6px;flex-wrap:wrap;margin-top:13px;}
.flow-outcomes span {font-size:.65rem;padding:2px 8px;border-radius:20px;
  border:1px solid #80c1ae;background:#ffffff10;color:#cceee3;}
.flow-outcomes span:last-child {border-color:#bcab75;color:#f2deb0;}
@media(max-width:1050px) and (min-width:761px){.decision-flow{gap:17px}.decision-flow .flow-stage{padding:15px 11px!important}.flow-stage:not(:last-child)::after{right:-15px;width:12px}.decision-flow h4{font-size:.86rem}.decision-flow p{font-size:.72rem!important}}
@media(max-width:760px){.decision-flow{grid-template-columns:1fr;gap:24px;margin-bottom:24px!important}
  .decision-flow .flow-stage{min-height:0;padding:18px 20px!important}.flow-card-top{margin-bottom:10px}
  .flow-stage:not(:last-child)::after{right:auto;left:calc(50% - 8px);top:auto;bottom:-17px;transform:rotate(90deg)}
  .decision-flow p{font-size:.83rem!important}.flow-outcomes{margin-top:10px}}
</style>'''
