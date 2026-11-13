from collections import Counter
import numpy as np
from src.constants.vocabulary import WHITELIST
from .dedupe import EMPTY_REASON,MIN_DF_REASON,group_documents,reason_counts

def build_tfidf(documents,min_df,dedupe=False):
 if not documents or not isinstance(min_df,int) or min_df<1:raise ValueError('语料为空或min_df不合法')
 if dedupe:return _build_grouped_tfidf(documents,min_df)
 allowed=set(WHITELIST);tokens=[[w for w in d['tokens_text'].split() if w in allowed] for d in documents]
 df=Counter(w for doc in tokens for w in set(doc));vocabulary=sorted(w for w,n in df.items() if n>=min_df)
 if not vocabulary:raise ValueError('白名单与min_df后词汇为空')
 n=len(documents);idf=np.asarray([np.log((1+n)/(1+df[w]))+1 for w in vocabulary]);lookup={w:i for i,w in enumerate(vocabulary)};X=np.zeros((n,len(vocabulary)))
 for i,doc in enumerate(tokens):
  for w,count in Counter(doc).items():
   if w in lookup:X[i,lookup[w]]=count*idf[lookup[w]]
  norm=np.linalg.norm(X[i])
  if norm:X[i]/=norm
 kept=np.flatnonzero(np.linalg.norm(X,axis=1)>0);excluded=[{'id':d['id'],'reference':d['reference'],'reason':'白名单后空词' if not tokens[i] else 'min_df后空词'} for i,d in enumerate(documents) if i not in set(kept)]
 return {'X':X[kept],'kept_indices':kept,'excluded':excluded,'vocabulary':vocabulary,'idf':idf.tolist(),'document_frequency':[df[w] for w in vocabulary],'idf_n_documents':n,'parameters':{'tokenizer':'str.split whitespace only; exact whitelist match','tf':'raw count','idf':'ln((1+n_documents)/(1+document_frequency))+1','norm':'L2 per document','min_df':min_df,'whitelist':list(WHITELIST),'empty_documents':'excluded from NMF; still counted in IDF n_documents'}}

def _build_grouped_tfidf(documents,min_df):
 grouped=group_documents(documents);groups=grouped['groups']
 if not groups:raise ValueError('白名单与min_df后词汇为空')
 df=Counter(item['term'] for group in groups for item in group['terms']);vocabulary=sorted(w for w,n in df.items() if n>=min_df)
 if not vocabulary:raise ValueError('白名单与min_df后词汇为空')
 n=len(groups);idf=np.asarray([np.log((1+n)/(1+df[w]))+1 for w in vocabulary]);term_index={w:i for i,w in enumerate(vocabulary)};X=np.zeros((n,len(vocabulary)))
 for group in groups:
  for item in group['terms']:
   if item['term'] in term_index:X[group['group_id'],term_index[item['term']]]=item['count']*idf[term_index[item['term']]]
  norm=np.linalg.norm(X[group['group_id']])
  if norm:X[group['group_id']]/=norm
 kept=np.flatnonzero(np.linalg.norm(X,axis=1)>0);kept_set={int(i) for i in kept}
 for group in groups:
  group['kept']=group['group_id'] in kept_set;group['excluded_reason']=None if group['kept'] else MIN_DF_REASON
 excluded=[]
 for document in documents:
  group_id=grouped['document_group'][document['id']]
  if group_id is None:excluded.append({'id':document['id'],'reference':document['reference'],'reason':EMPTY_REASON})
  elif group_id not in kept_set:excluded.append({'id':document['id'],'reference':document['reference'],'reason':MIN_DF_REASON})
 content_documents=sum(1 for terms in grouped['document_terms'] if terms)
 deduplication={'mode':'term_frequency','algorithm':'whitelist-multiset-v1','original_documents':len(documents),'content_documents':content_documents,'group_count':len(groups),'duplicate_documents':content_documents-len(groups),'empty_documents':len(grouped['empty_documents']),'effective_groups':int(len(kept)),'excluded_group_count':len(groups)-int(len(kept)),'excluded_document_count':len(excluded),'excluded_reasons':reason_counts(documents,groups,grouped['document_group'],kept_set)}
 parameters={'tokenizer':'str.split whitespace only; exact whitelist match','tf':'raw term count per distinct group','idf':'ln((1+n_groups)/(1+group_document_frequency))+1','norm':'L2 per group','min_df':min_df,'whitelist':list(WHITELIST),'grouping':'whitelist-filtered term/count multiset; word order ignored; copies share one row','empty_documents':'whitelist-empty documents listed individually; not grouped, not counted in group df/IDF/fit','dedupe':True}
 return {'X':X[kept],'kept_indices':kept,'excluded':excluded,'vocabulary':vocabulary,'idf':idf.tolist(),'document_frequency':[df[w] for w in vocabulary],'idf_n_documents':n,'parameters':parameters,'groups':groups,'document_group':grouped['document_group'],'deduplication':deduplication}
