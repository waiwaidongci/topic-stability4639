from collections import Counter
from src.constants.vocabulary import WHITELIST

EMPTY_REASON='白名单后空词'
MIN_DF_REASON='min_df后空词'
REASON_ORDER=(EMPTY_REASON,MIN_DF_REASON)

def whitelist_terms(tokens_text,allowed=WHITELIST):
 allowed_set=set(allowed)
 return [word for word in tokens_text.split() if word in allowed_set]

def group_key(terms):
 return tuple(sorted(Counter(terms).items()))

def group_documents(documents,allowed=WHITELIST):
 groups_by_key={}
 document_terms=[whitelist_terms(document['tokens_text'],allowed) for document in documents]
 for document,terms in zip(documents,document_terms):
  if not terms:continue
  key=group_key(terms)
  group=groups_by_key.setdefault(key,{'terms':[{'term':term,'count':count} for term,count in key],'members':[]})
  group['members'].append(document)
 groups=[]
 for index,key in enumerate(sorted(groups_by_key)):
  source=groups_by_key[key];members=sorted(source['members'],key=lambda document:document['reference'])
  groups.append({'group_id':index,'group_code':f'G{index+1:02d}','terms':source['terms'],'member_ids':[document['id'] for document in members],'member_references':[document['reference'] for document in members],'member_count':len(members)})
 mapping={document['id']:None for document in documents}
 for group in groups:
  for document_id in group['member_ids']:mapping[document_id]=group['group_id']
 return {'groups':groups,'document_group':mapping,'document_terms':document_terms,'empty_documents':[document for document,terms in zip(documents,document_terms) if not terms]}

def reason_counts(documents,groups,document_group,kept_group_ids):
 kept=set(kept_group_ids);counts={reason:{'document_count':0,'group_count':0} for reason in REASON_ORDER}
 for document in documents:
  group_id=document_group[document['id']]
  if group_id is None:counts[EMPTY_REASON]['document_count']+=1
  elif group_id not in kept:counts[MIN_DF_REASON]['document_count']+=1
 for group in groups:
  if group['group_id'] not in kept:counts[MIN_DF_REASON]['group_count']+=1
 return [{'reason':reason,**counts[reason]} for reason in REASON_ORDER if counts[reason]['document_count']]
