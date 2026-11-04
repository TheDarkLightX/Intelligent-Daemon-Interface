import {gunzipSync} from 'node:zlib';
// One final, source-grouped test of document recovery, not answer truthfulness.
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {Agent} from './web/engine.mjs';
import {terms} from './web/runtime.mjs';
const read=name=>readFileSync(new URL(name,import.meta.url));
const json=name=>JSON.parse(read(name));
const bundle=json('web/model.json'),knowledge=JSON.parse(gunzipSync(read('web/knowledge.json.gz')));
if(bundle.metadata.provisional)throw Error('Freeze the final model before opening this test');
const agent=await Agent.create(bundle,json('web/cards.json'),knowledge,read('web/evaluator.wasm'));
const tests=knowledge.records.filter(r=>r.split==='test');
const report={scope:'The search collection contains the held-out documents. Queries are their original questions or every fifth lexical term removed. Correct means the exact original record ID, not semantic equivalence, factual accuracy, or open-chat competence. No parameters are fitted here.',checkpoint:bundle.metadata.checkpointSha256,source_sha256:Object.fromEntries(['web/engine.mjs','web/runtime.mjs','evaluate_retrieval.mjs','web/knowledge.json.gz'].map(n=>[n,createHash('sha256').update(read(n)).digest('hex')])),conditions:{}};
for(const condition of ['original','drop_every_fifth_term']) {
  const started=performance.now(),counts={rows:tests.length,bm25_top1:0,bm25_top12:0,reciprocal_rank_sum:0,direct_exact_record:0,confirmed_exact_record:0,wrong_direct_record:0,wrong_pending_record:0,other_routes:0},routes={},examples=[];
  for(const record of tests) {
    const query=condition==='original'?record.q:terms(record.q).filter((_,i)=>i%5!==4).join(' ');
    const ranked=agent.index.search(query),position=ranked.findIndex(m=>knowledge.records[m.i].id===record.id);
    if(position===0)counts.bm25_top1++;
    if(position>=0){counts.bm25_top12++;counts.reciprocal_rank_sum+=1/(position+1);}
    agent.context=null;const response=agent.reply(query);routes[response.kind]=(routes[response.kind]||0)+1;
    if(response.kind==='retrieved') {if(response.recordId===record.id)counts.direct_exact_record++;else counts.wrong_direct_record++;}
    else if(response.kind==='clarify') {if(agent.context.pending.id===record.id)counts.confirmed_exact_record++;else counts.wrong_pending_record++;}
    else counts.other_routes++;
    if(examples.length<12&&(response.kind==='retrieved'&&response.recordId!==record.id||response.kind==='clarify'&&agent.context.pending.id!==record.id))examples.push({expected:record.id,query,kind:response.kind,actual:response.recordId||agent.context.pending.id});
  }
  report.conditions[condition]={...counts,routes,top1:counts.bm25_top1/tests.length,top12:counts.bm25_top12/tests.length,mrr12:counts.reciprocal_rank_sum/tests.length,direct_exact_rate:counts.direct_exact_record/tests.length,direct_or_confirmable_exact_rate:(counts.direct_exact_record+counts.confirmed_exact_record)/tests.length,seconds:Math.round((performance.now()-started)/10)/100,examples};
}
const out=new URL('evidence/retrieval-final/',import.meta.url);mkdirSync(out,{recursive:true});writeFileSync(new URL('report.json',out),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report));
