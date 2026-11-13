import numpy as np
from .tfidf import build_tfidf
from .factorization import fit_matrix
from .alignment import align_topics
from .fingerprint import fingerprint

INTERPRETATION='K is preselected, not optimized by decreasing reconstruction error. Relative activations are not posterior probabilities. Stability measures seed sensitivity, not semantic truth; editorial names are human input. All repeats refit independently with random init.'
REASON_ORDER=('白名单后空词','min_df后空词')

def excluded_reasons(excluded):
 counts={reason:0 for reason in REASON_ORDER}
 for entry in excluded:counts[entry['reason']]=counts.get(entry['reason'],0)+1
 return [{'reason':reason,'count':count} for reason,count in counts.items() if count]

def analyze(corpus,documents,config):
 documents=sorted(documents,key=lambda d:d['reference'])
 if len({d['reference'] for d in documents})!=len(documents):raise ValueError('文档reference重复')
 if not isinstance(config['repeats'],int) or not 5<=config['repeats']<=10:raise ValueError('初始化重复数必须5..10')
 if not isinstance(config['seed'],int) or not 0<=config['seed']<=1000000000:raise ValueError('seed范围不合法')
 dedupe=bool(config.get('dedupe',False));data=build_tfidf(documents,config['min_df'],dedupe=dedupe);X=data['X'];k=config['k'];fits=[]
 for index in range(config['repeats']):
  seed=config['seed']+index
  fit=fit_matrix(X,k,seed,config['max_iter'],config['tolerance']);fits.append(fit)
 reference=fits[0]['H'];replicates=[]
 for index,fit in enumerate(fits):
  match=align_topics(reference,fit['H']);permutation=match['permutation'];replicates.append({'index':index,'seed':config['seed']+index,'iterations':fit['iterations'],'reconstruction_error':fit['reconstruction_error'],'relative_error':fit['relative_error'],'W':fit['W'].tolist(),'H':fit['H'].tolist(),'aligned_W':fit['W'][:,permutation].tolist(),'aligned_H':fit['H'][permutation].tolist(),'alignment':match})
 stability=np.asarray([r['alignment']['similarities'] for r in replicates[1:]])
 profiles=[]
 for topic in range(k):
  weights=reference[topic];order=np.argsort(-weights,kind='stable');profiles.append({'topic_index':topic,'label':f'待命名主题{topic+1:02d}','weights':[{'term':data['vocabulary'][int(i)],'weight':float(weights[i])} for i in order],'mean_similarity':float(stability[:,topic].mean()),'similarities':stability[:,topic].tolist()})
 baseW=fits[0]['W']
 fingerprint_config={key:value for key,value in config.items() if not (key=='dedupe' and not value)}
 result={'fingerprint':fingerprint(corpus,documents,fingerprint_config),'dedupe':dedupe,'mode':'grouped' if dedupe else 'legacy','k':k,'repeats':config['repeats'],'reference_repeat':0,'reference_seed':config['seed'],'vocabulary':data['vocabulary'],'idf':data['idf'],'document_frequency':data['document_frequency'],'tfidf_parameters':data['parameters'],'idf_n_documents':data['idf_n_documents'],'original_documents':len(documents),'excluded_documents':data['excluded'],'excluded_reasons':excluded_reasons(data['excluded']),'profiles':profiles,'replicates':replicates,'mean_alignment_similarity':float(stability.mean()),'baseline_relative_error':fits[0]['relative_error'],'interpretation':INTERPRETATION}
 if dedupe:
  groups=data['groups'];kept_groups=[group for group in groups if group['kept']]
  for group_index,group in enumerate(kept_groups):
   activation=baseW[group_index];total=float(activation.sum())
   group['activation']=activation.tolist();group['relative_activation']=(activation/total).tolist() if total>1e-14 else None
  document_topics=[{'group_id':group['group_id'],'group_code':group['group_code'],'member_count':group['member_count'],'member_ids':group['member_ids'],'member_references':group['member_references'],'terms':group['terms'],'reference':' / '.join(group['member_references']),'title':f"{group['group_code']} · {group['member_count']}篇同词频",'activation':group['activation'],'relative_activation':group['relative_activation']} for group in kept_groups]
  document_groups=[]
  for document in documents:
   group_id=data['document_group'][document['id']]
   entry={'id':document['id'],'reference':document['reference'],'title':document['title'],'group_id':group_id,'group_code':None,'kept':False,'excluded_reason':None,'relative_activation':None}
   if group_id is None:entry['excluded_reason']='白名单后空词'
   else:
    group=groups[group_id];entry.update({'group_code':group['group_code'],'kept':group['kept'],'excluded_reason':group['excluded_reason'],'relative_activation':group.get('relative_activation')})
   document_groups.append(entry)
  result.update({'effective_documents':sum(group['member_count'] for group in kept_groups),'effective_groups':len(kept_groups),'group_count':len(groups),'kept_document_ids':[document_id for group in kept_groups for document_id in group['member_ids']],'kept_group_ids':[group['group_id'] for group in kept_groups],'document_groups':document_groups,'document_topics':document_topics,'deduplication':data['deduplication'],'dedupe_parameters':data['parameters'],'interpretation':INTERPRETATION+' Term-frequency duplicates share one NMF row, so copies of a sign do not add weight to its terms. Groups are the whitelist-filtered term/count multiset; word order does not matter and whitelist-empty documents stay individual.'})
 else:
  kept=[documents[int(i)] for i in data['kept_indices']];scores=[]
  for i,doc in enumerate(kept):
   activation=baseW[i];total=float(activation.sum());scores.append({'id':doc['id'],'reference':doc['reference'],'title':doc['title'],'activation':activation.tolist(),'relative_activation':(activation/total).tolist() if total>1e-14 else None})
  result.update({'effective_documents':len(kept),'effective_groups':len(kept),'kept_document_ids':[d['id'] for d in kept],'document_topics':scores,'interpretation':INTERPRETATION+' Legacy document-frequency mode: each original document is one NMF row, so repeated sign copies are counted separately.'})
 return result
