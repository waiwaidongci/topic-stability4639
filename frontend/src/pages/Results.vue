<script setup lang="ts">
import {ref,computed} from 'vue'
import type {Run,Profile,User} from '../types/domain'
import {researchApi} from '../api/research'
import {useAsync} from '../hooks/useAsync'
import PageTitle from '../components/common/PageTitle.vue'
import Badge from '../components/common/Badge.vue'
import Empty from '../components/common/Empty.vue'
import Notice from '../components/common/Notice.vue'
import Table from '../components/common/Table.vue'
import TopicTerms from '../components/TopicTerms.vue'
import RunSummary from '../components/RunSummary.vue'
import AlignmentGrid from '../components/AlignmentGrid.vue'
import {pct,dec} from '../utils/format'

const props=defineProps<{runs:Run[];profiles:Profile[];selectedId:number;user:User;reload:()=>Promise<any>}>()
const emit=defineEmits<{select:[run:Run]}>()
const run=computed(()=>props.runs.find(r=>r.id===props.selectedId)||props.runs[0])
const topics=computed(()=>props.profiles.filter(p=>p.run_id===run.value?.id))
const focusId=ref(0)
const focus=computed(()=>topics.value.find(p=>p.id===focusId.value)||topics.value[0])
const label=ref('人工主题A')
const editorNote=ref('结合词权和代表文档人工命名，非模型自动语义结论。')
const noteName=ref('展签主题编辑说明')
const interpretation=ref('相对激活用于代表文档阅读；对齐稳定仅反映初始化敏感性，不等于语义真值。')
const nextStep=ref('逐篇阅读混合主题摘要，与策展编辑核对命名和改稿范围。')
const {task,act}=useAsync()

const grouped=computed(()=>run.value?.result.mode==='grouped'&&!!run.value.result.document_groups)
const dedup=computed(()=>run.value?.result.deduplication)
const groupTopicRows=computed(()=>run.value?.result.document_topics?.slice(0,20)||[])
const documentRows=computed(()=>run.value?.result.document_groups?.slice(0,40)||[])

function groupTerms(row:{terms?:{term:string;count:number}[]}){
 return (row.terms||[]).slice(0,8).map(item=>item.term+'×'+item.count).join(' · ')
}
</script>

<template>
  <template v-if="run">
    <PageTitle index="04 / WORDS & STABILITY" title="读词权，也读初始化的变化。" description="主题编号只是参照分解的位置，主题名由编辑给出；相对激活权重不称概率后验。按有效词频分组后，重复副本不改变词权与拟合。"/>
    <Notice :error="task.error||run.error" :message="task.message"/>
    <div class="run-selector">
      <select aria-label="选择运行" :value="run.id" @change="emit('select',runs.find(r=>r.id===Number(($event.target as HTMLSelectElement).value))!)">
        <option v-for="r in runs" :key="r.id" :value="r.id">{{r.name}}</option>
      </select>
      <Badge :status="run.status"/>
      <small>SHA-256 {{run.result.fingerprint?.slice(0,26)||'等待计算'}}…</small>
    </div>
    <template v-if="topics.length">
      <RunSummary :run="run"/>
      <div class="topic-layout">
        <section>
          <div class="topic-tabs">
            <button v-for="p in topics" :key="p.id" :class="{active:p.id===focus?.id}" @click="focusId=p.id">{{p.topic_index+1}} / {{p.label}}</button>
          </div>
          <TopicTerms v-if="focus" :profile="focus"/>
        </section>
        <aside class="normalization-note">
          <small>SCALE / RECONSTRUCTION</small>
          <h2>{{dec(run.result.baseline_relative_error)}}</h2>
          <p>参照seed {{run.result.reference_seed}}的相对Frobenius误差，仅解释固定语料与K下的拟合，不作跨K自动最优或语义质量分数。</p>
          <p v-if="dedup">原篇{{dedup.original_documents}}篇 → 独立组{{dedup.group_count}}组（重复副本{{dedup.duplicate_documents}}篇不另计），有效组{{dedup.effective_groups}}组；IDF与拟合均按组计数。</p>
          <p v-else>旧口径：每篇原文各占一行，重复副本会各自计入词频与拟合。有效{{run.result.effective_documents}}篇。</p>
          <p>H每行词权和为1；W乘相应范数，归一前后WH一致。排除原文{{run.result.excluded_documents?.length}}篇。</p>
          <p>平均匹配余弦来自其余{{run.config.repeats-1}}次独立重拟合，不含参照自身1.0。</p>
        </aside>
      </div>
      <form v-if="focus&&user.role!=='viewer'" class="editor naming-editor" @submit.prevent="act(async()=>{await researchApi.name(focus!.id,{label,editor_note:editorNote});await reload()},'人工主题名已保存')">
        <h2>编辑命名 / 当前主题{{focus.topic_index+1}}</h2>
        <div class="form-grid">
          <label>人工主题名称<input v-model="label" required maxlength="100"/></label>
          <label>命名依据<input v-model="editorNote" maxlength="500"/></label>
        </div>
        <button class="primary" :disabled="task.busy">保存人工命名 →</button>
      </form>
      <AlignmentGrid :run="run"/>
      <template v-if="grouped">
        <div class="section-title">
          <h2>独立组词频 / 副本合并计数</h2>
          <small>白名单后词项及次数相同即同组，词序不影响；参照分解前20有效组</small>
        </div>
        <Table :headers="['组 / 组成员','词项 × 次数',...topics.map(p=>p.label)]">
          <tr v-for="g in groupTopicRows" :key="g.group_id">
            <td><small>{{g.group_code}} · {{g.member_count}}篇</small><strong>{{g.member_references.join(' / ')}}</strong></td>
            <td><small class="token-text">{{groupTerms(g)}}</small></td>
            <td v-for="p in topics" :key="p.id">{{g.relative_activation?pct(g.relative_activation[p.topic_index]):'未激活'}}</td>
          </tr>
        </Table>
        <div class="section-title">
          <h2>原篇回查 / 所属组与主题激活</h2>
          <small>展示前40篇，计算用全部{{run.result.document_groups?.length}}篇</small>
        </div>
        <Table :headers="['原文 / 标题','所属组','排除原因',...topics.map(p=>p.label)]">
          <tr v-for="d in documentRows" :key="d.id">
            <td><small>{{d.reference}}</small><strong>{{d.title}}</strong></td>
            <td>{{d.group_code||'—'}}</td>
            <td>{{d.excluded_reason||'—'}}</td>
            <td v-for="p in topics" :key="p.id">{{d.relative_activation?pct(d.relative_activation[p.topic_index]):'—'}}</td>
          </tr>
        </Table>
        <p class="excluded-note">排除原因：{{dedup?.excluded_reasons?.map(r=>r.reason+' '+r.document_count+'篇'+(r.group_count?' / '+r.group_count+'组':'')).join('；')||'无'}}。分组只合并白名单后词项及次数完全相同的展签；空词单列，不参与词频、IDF与拟合。</p>
      </template>
      <template v-else>
        <div class="section-title">
          <h2>文档相对激活 / 非后验概率</h2>
          <small>旧口径逐篇计数；参照分解前20有效文档</small>
        </div>
        <Table :headers="['文档 / 标题',...topics.map(p=>p.label)]">
          <tr v-for="d in groupTopicRows" :key="d.id">
            <td><small>{{d.reference}}</small><strong>{{d.title}}</strong></td>
            <td v-for="p in topics" :key="p.id">{{d.relative_activation?pct(d.relative_activation[p.topic_index]):'未激活'}}</td>
          </tr>
        </Table>
        <p class="excluded-note">排除文档：{{run.result.excluded_documents?.map(d=>d.reference+' '+d.reason).join('；')||'无'}}。</p>
      </template>
      <div class="section-title">
        <h2>冻结词汇、idf与排除记录</h2>
        <small>idf文档总量{{run.result.idf_n_documents}}，{{grouped?'按独立组计':'按原文计'}}，包含空词</small>
      </div>
      <div class="vocabulary-grid">
        <div v-for="(word,i) in run.result.vocabulary" :key="word"><strong>{{word}}</strong><small>df {{run.result.document_frequency?.[i]}} / idf {{dec(run.result.idf?.[i])}}</small></div>
      </div>
      <p class="excluded-note">参数：raw TF，ln((1+n)/(1+df))+1，L2，min_df={{run.config.min_df}}，{{grouped?'按独立组计数':'按原文计数'}}。</p>
      <form v-if="focus&&user.role!=='viewer'" class="editor note-editor" @submit.prevent="act(async()=>{await researchApi.note({run_id:run!.id,name:noteName,profile_ids:[focus!.id],interpretation,next_step:nextStep});await reload()},'编辑说明已建立，等待负责人审阅')">
        <small>NEXT / EDITORIAL RECORD</small>
        <h2>以当前人工主题建立说明</h2>
        <label>说明名称<input v-model="noteName" required maxlength="100"/></label>
        <label>编辑解释<textarea v-model="interpretation" rows="3" required maxlength="700"/></label>
        <label>下一步阅读与改稿<input v-model="nextStep" required maxlength="500"/></label>
        <p>建立时冻结主题名、词权与稳定性快照，后续命名修订不会改写已审阅交付。</p>
        <button class="primary" :disabled="task.busy">建立编辑说明 →</button>
      </form>
    </template>
    <Empty v-else title="等待worker或检查失败原因" help="空词、常量/低秩、不可行K或任一未收敛会明确failed；调整输入与参数后重新运行。"/>
  </template>
  <Empty v-else title="尚无主题册" help="从主题配置选择K与独立初始化次数。"/>
</template>
