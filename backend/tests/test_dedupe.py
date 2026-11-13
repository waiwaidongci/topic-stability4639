import unittest
import numpy as np
from src.services.analysis import analyze
from src.services.tfidf import build_tfidf

CORPUS={'id':1,'name':'测试语料','filename':'test.jsonl','description':''}

def document(index,reference,tokens):
 return {'id':index,'reference':reference,'title':'展签'+reference,'tokens_text':' '.join(tokens)}

def corpus_documents():
 return [
  document(1,'D-01',['陶土','陶土','工艺','器皿','釉色','烧成']),
  document(2,'D-02',['烧成','釉色','器皿','工艺','陶土','陶土']),
  document(3,'D-03',['山水','河流','林间','季节','岩石','地貌']),
  document(4,'D-04',['地貌','岩石','季节','林间','河流','山水']),
  document(5,'D-05',['陶土','工艺','器皿','窑','釉色','烧成']),
  document(6,'D-06',['陌词','占位']),
  document(7,'D-07',[]),
  document(8,'D-08',['研究']),
 ]

CONFIG={'k':2,'repeats':5,'seed':7,'min_df':1,'max_iter':2000,'tolerance':.0001}

class GroupingTests(unittest.TestCase):
 def test_order_does_not_matter_and_copies_share_group(self):
  docs=corpus_documents();groups=build_tfidf(docs,1,dedupe=True)['groups']
  by_reference={reference:group['group_code'] for group in groups for reference in group['member_references']}
  self.assertEqual(by_reference['D-01'],by_reference['D-02'])
  self.assertEqual(by_reference['D-03'],by_reference['D-04'])
  self.assertEqual(len(groups),4)
  members={group['group_code']:group['member_count'] for group in groups}
  self.assertEqual(sorted(members.values()),[1,1,2,2])

 def test_empty_whitelist_documents_are_not_grouped(self):
  docs=corpus_documents();data=build_tfidf(docs,1,dedupe=True)
  references={group['group_code']:group['member_references'] for group in data['groups']}
  self.assertNotIn('D-06',[reference for refs in references.values() for reference in refs])
  self.assertNotIn('D-07',[reference for refs in references.values() for reference in refs])
  self.assertEqual([entry['reason'] for entry in data['excluded']],['白名单后空词']*2)

 def test_duplicate_copies_do_not_change_counts_or_fit(self):
  docs=corpus_documents();base=analyze(CORPUS,docs,CONFIG|{'dedupe':True})
  copies=[{'id':100+index,'reference':'COPY-'+doc['reference'],'title':doc['title'],'tokens_text':doc['tokens_text']} for index,doc in enumerate(docs) if doc['tokens_text'].split()]
  expanded=analyze(CORPUS,docs+copies,CONFIG|{'dedupe':True})
  self.assertEqual(expanded['vocabulary'],base['vocabulary'])
  self.assertEqual(expanded['idf'],base['idf'])
  self.assertEqual(expanded['document_frequency'],base['document_frequency'])
  self.assertEqual(expanded['idf_n_documents'],base['idf_n_documents'])
  self.assertEqual(expanded['baseline_relative_error'],base['baseline_relative_error'])
  self.assertEqual([r['relative_error'] for r in expanded['replicates']],[r['relative_error'] for r in base['replicates']])
  self.assertEqual([group['relative_activation'] for group in expanded['document_topics']],[group['relative_activation'] for group in base['document_topics']])
  self.assertEqual([group['group_code'] for group in expanded['document_topics']],[group['group_code'] for group in base['document_topics']])
  self.assertEqual(expanded['deduplication']['group_count'],base['deduplication']['group_count'])
  content_copies=sum(1 for doc in copies if doc['reference']!='COPY-D-06')
  self.assertEqual(expanded['deduplication']['duplicate_documents'],base['deduplication']['duplicate_documents']+content_copies)
  self.assertEqual(expanded['deduplication']['duplicate_documents'],expanded['deduplication']['content_documents']-expanded['deduplication']['group_count'])

 def test_word_order_does_not_change_result(self):
  docs=corpus_documents();base=analyze(CORPUS,docs,CONFIG|{'dedupe':True})
  shuffled=[dict(doc,tokens_text=' '.join(reversed(doc['tokens_text'].split()))) for doc in docs]
  reshuffled=analyze(CORPUS,shuffled,CONFIG|{'dedupe':True})
  self.assertEqual(reshuffled['baseline_relative_error'],base['baseline_relative_error'])
  self.assertEqual(reshuffled['deduplication'],base['deduplication'])
  self.assertNotEqual(reshuffled['fingerprint'],base['fingerprint'])

 def test_min_df_counts_groups_not_copies(self):
  docs=[document(1,'R-01',['陶土','工艺']),document(2,'R-02',['工艺','陶土']),document(3,'R-03',['山水','河流']),document(4,'R-04',['河流','山水']),document(5,'R-05',['陶土','工艺','山水'])]
  grouped=build_tfidf(docs,2,dedupe=True)
  self.assertEqual(grouped['document_frequency'],sorted(grouped['document_frequency']))
  self.assertEqual(dict(zip(grouped['vocabulary'],grouped['document_frequency'])),{'山水':2,'工艺':2,'陶土':2})
  legacy=build_tfidf(docs,2,dedupe=False)
  self.assertEqual(dict(zip(legacy['vocabulary'],legacy['document_frequency'])),{'山水':3,'工艺':3,'河流':2,'陶土':3})
  self.assertEqual(legacy['idf_n_documents'],5)
  self.assertEqual(grouped['idf_n_documents'],3)

 def test_every_original_document_reports_group_and_activation(self):
  docs=corpus_documents();result=analyze(CORPUS,docs,CONFIG|{'dedupe':True})
  lookup={entry['id']:entry for entry in result['document_groups']}
  self.assertEqual(len(lookup),len(docs))
  kept_references={reference for group in result['document_topics'] for reference in group['member_references']}
  for doc in docs:
   entry=lookup[doc['id']]
   self.assertEqual(entry['reference'],doc['reference'])
   if entry['kept']:
    self.assertIsNotNone(entry['group_code'])
    self.assertIn(doc['reference'],kept_references)
    self.assertIsNotNone(entry['relative_activation'])
    self.assertAlmostEqual(sum(entry['relative_activation']),1.0,places=12)
   else:
    self.assertIn(entry['excluded_reason'],('白名单后空词','min_df后空词'))

 def test_legacy_mode_keeps_per_document_rows(self):
  docs=corpus_documents();result=analyze(CORPUS,docs,CONFIG)
  self.assertFalse(result['dedupe'])
  self.assertEqual(result['mode'],'legacy')
  self.assertEqual(result['idf_n_documents'],len(docs))
  self.assertEqual(result['effective_documents'],len(result['document_topics']))
  self.assertNotIn('document_groups',result)
  self.assertNotIn('dedupe',result['fingerprint'])
  self.assertEqual(result['effective_documents'],6)

if __name__=='__main__':
 unittest.main()
