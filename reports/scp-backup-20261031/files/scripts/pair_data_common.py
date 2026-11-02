"""Shared semantic conversion for the authorized candidate-matching experiment."""
import json
import re

DOMAINS = ['knowledge', 'language', 'retrieval', 'tools', 'arts']
NO_TRAIN = re.compile(r'(^|[/_-])(gpqa|mmlu[-_]?pro|bbh|big[-_]?bench|hle|simplebench|cruxeval|cladder|musr|blimp|phishnchips|sata[-_]?bench)([/_-]|$)', re.I)
ARTS = re.compile(r'humicroedit|humor|humour|joke|funny|poem|poetry|music|chord|cfcolor|habermas|new.?yorker|helpsteer|ultrafeedback|hh.?rlhf|preference|rewardbench|argument.?quality|aesthetic|would.?you.?rather|stanford.*shp', re.I)
TOOLS = re.compile(r'tool|api.?bank|function.?call|glaive|gorilla|xlam|routing|router|workflow|state.?machine|home.?assistant|when2call|browser|json.?schema|business.?policy', re.I)
RETRIEVAL = re.compile(r'banking|clinc|massive|intent|ag.?news|dbpedia|topic|retrieval|rerank|relevance|esci|ms.?marco|hover|spam|phish|schema.?guided|yahoo.?answers.?topic|masakhanews|amazon.*category', re.I)
LANGUAGE = re.compile(r'hellaswag|anli|mnli|snli|wanli|docnli|entail|nli|sentiment|sst|sarcas|irony|vast|stance|contract|nli4ct|ragtruth|ground|tweet|emotion|hate|toxicity|review|grammar|cola|paraphrase|paws|qqp|mrpc|similarity|sts', re.I)
KNOWLEDGE = re.compile(r'gsm|math|arc|science|sciq|openbook|commonsense|winogrande|winograd|copa|piqa|boolq|logic|reason|causal|chess|code|mbpp|crux|program|medical|medmc|pubmed|mmlu|qasc|qa|theory.?of.?mind|temporal|spatial|count|arithmetic', re.I)
LOGIC_KNOWLEDGE = re.compile(r'reclor|logiqa|logic|proofwriter|ruletaker|folio|clutrr|boardgame|babi|strategyqa|causal|chess|deduct|abduct|ethics|moral.?reason', re.I)
REFERENCES = re.compile(r'(all|none|both|neither).{0,32}(above|other|option|answer|criteri)|(?:option|answer|choice)\s+[A-Z](?:\b|\))', re.I)
RELATIVE = re.compile(r'better|preferred|preferable|funnier|more helpful|more appropriate|which.*(?:prefer|quality)', re.I)


def domain_for(source):
    if ARTS.search(source): return 'arts'
    if TOOLS.search(source): return 'tools'
    if RETRIEVAL.search(source): return 'retrieval'
    if LOGIC_KNOWLEDGE.search(source): return 'knowledge'
    if LANGUAGE.search(source): return 'language'
    if KNOWLEDGE.search(source): return 'knowledge'
    return 'language'


def text(value):
    if value is None: return ''
    if isinstance(value, str): return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def criteria_context(question, options):
    if not (any(REFERENCES.search(option) for option in options) or (len(options) <= 8 and RELATIVE.search(question))):
        return ''
    labels = [chr(65 + i) if i < 26 else str(i + 1) for i in range(len(options))]
    return '\n'.join(f'{label}: {option}' for label, option in zip(labels, options))


def render_pair(state, question, candidate, criteria=''):
    prefix = 'Decide whether the candidate is a correct match.\nInput:\n' + text(state)
    if question: prefix += '\nQuestion:\n' + text(question)
    if criteria: prefix += '\nAvailable criteria:\n' + criteria
    return prefix + '\nCandidate:\n', text(candidate)
