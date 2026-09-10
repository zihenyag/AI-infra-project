"""由已校验结果生成中文报告源稿、原生图表artifact和附加传输图。"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from analyze_v100_experiment import collect, quantile
from v100_chart_queries import query_raw

LABELS = {'random_2k_q1': '随机2K', 'prefix_2k_q1': '共享2K', 'pressure_4k_q1': '压力4K',
          'prefix_2k_q025': '2K/QPS0.25', 'prefix_2k_q05': '2K/QPS0.5',
          'prefix_1k_q1': '共享1K', 'prefix_4k_q1': '共享4K'}
ORDER = list(LABELS)


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def make_report(root):
    report, raw = collect(root / 'raw')
    assert report['complete']
    rows = report['rows']
    lookup = {(r['baseline'], r['point']): r for r in rows}
    get = lambda b, p: lookup[b, p]
    s1, s2 = get('S1', 'pressure_4k_q1'), get('S2', 'pressure_4k_q1')
    pct = lambda value: f'{value:.2%}'
    delta = lambda a,b: (a-b)/b
    values = {
        'pressure_s1_hit': pct(s1['cache_hit_ratio']), 'pressure_s2_hit': pct(s2['cache_hit_ratio']),
        'pressure_ttft_drop': pct(-delta(s2['ttft_ms_p50'], s1['ttft_ms_p50'])),
        'pressure_p95_rise': pct(delta(s2['ttft_ms_p95'], s1['ttft_ms_p95'])),
        'pressure_p99_rise': pct(delta(s2['ttft_ms_p99'], s1['ttft_ms_p99'])),
        'pressure_output_gain': pct(delta(s2['output_token_throughput'], s1['output_token_throughput'])),
        'random_ttft_rise': pct(delta(get('S2','random_2k_q1')['ttft_ms_p50'],get('S1','random_2k_q1')['ttft_ms_p50'])),
        'prefix_ttft_rise': pct(delta(get('S2','prefix_2k_q1')['ttft_ms_p50'],get('S1','prefix_2k_q1')['ttft_ms_p50'])),
        'prefix_s1_hit': pct(get('S1','prefix_2k_q1')['cache_hit_ratio']),
        'prefix_s2_hit': pct(get('S2','prefix_2k_q1')['cache_hit_ratio']),
        'total_d2h_gb': f"{sum(r['gpu_to_cpu_bytes'] for r in rows)/1e9:.3f}",
        'total_h2d_gb': f"{sum(r['cpu_to_gpu_bytes'] for r in rows)/1e9:.3f}",
        'pressure_d2h_s': f"{s2['gpu_to_cpu_host_call_s']:.3f}",
        'pressure_h2d_s': f"{s2['cpu_to_gpu_host_call_s']:.3f}",
        'pressure_d2h_bw': f"{s2['gpu_to_cpu_GBps_by_host_call_s']:.2f}",
        'pressure_h2d_bw': f"{s2['cpu_to_gpu_GBps_by_host_call_s']:.2f}",
    }
    for b,r in [('s1',s1),('s2',s2)]:
        for p in ('p50','p95'):
            values[f'pressure_{b}_{p}'] = f"{r[f'ttft_ms_{p}']:.2f}"
    for b in ('S0','S1','S2'):
        values[f'wait_{b.lower()}'] = f"{get(b,'prefix_2k_q1')['client_wait_ms_mean']/1000:.2f}"
    manifest = json.loads((root/'raw/manifest.json').read_text())
    values['matrix_table'] = table(['测点','模式','目标/实际平均ISL','QPS','请求数','前缀组'],[
        [p['name'],p['mode'],f"{p['isl']} / {get('S0',p['name'])['actual_isl_mean']:.2f}",p['qps'],p['n'],p['groups']]
        for p in manifest['points']])
    core = [get(b,p) for p in ORDER[:3] for b in ('S0','S1','S2')]
    values['core_table'] = table(['负载','组','TTFT p50/p95(ms)','ITL p50(ms)','E2E p50(s)','输出tokens/s','总复用率'],[
        [LABELS[r['point']],r['baseline'],f"{r['ttft_ms_p50']:.2f}/{r['ttft_ms_p95']:.2f}",
         f"{r['itl_ms_p50']:.2f}",f"{r['e2e_ms_p50']/1000:.2f}",f"{r['output_token_throughput']:.2f}",pct(r['cache_hit_ratio'])] for r in core])
    transfers = []
    for p in ORDER:
        r = get('S2',p)
        for d,label in [('gpu_to_cpu','GPU→CPU'),('cpu_to_gpu','CPU→GPU')]:
            transfers.append({'point':p,'label':LABELS[p],'direction':label,'bytes':r[d+'_bytes'],
                'GB':r[d+'_bytes']/1e9,'time_s':r[d+'_host_call_s'],'GBps':r[d+'_GBps_by_host_call_s'],
                'operations':r[d+'_operations'],'size_p50_MiB':(r[d+'_size_bytes_p50']/1024**2) if r[d+'_size_bytes_p50'] is not None else None,
                'size_p95_MiB':(r[d+'_size_bytes_p95']/1024**2) if r[d+'_size_bytes_p95'] is not None else None,
                'n':r['completed'],'host_reuse_tokens':r['host_reuse_tokens']})
    fmt = lambda v: '不可用' if v is None else f'{v:.3f}'
    values['offload_table'] = table(['测点','方向','GB','调用总秒数','GB/s','操作数','单次MiB p50/p95'],[
        [x['label'],x['direction'],fmt(x['GB']),fmt(x['time_s']),fmt(x['GBps']),x['operations'],
         f"{fmt(x['size_p50_MiB'])}/{fmt(x['size_p95_MiB'])}"] for x in transfers])
    template = Path(__file__).with_name('v100_report_template.md').read_text()
    for key, value in values.items():
        template = template.replace('{{'+key+'}}', value)
    assert not re.findall(r'\{\{(?!chart:)[^}]+\}\}',template)
    datasets = {}
    def clean(r):
        return {**{k:v for k,v in r.items() if k not in ('source','gpu_uuid')},
                'label':LABELS[r['point']], 'line_style':{'S0':'solid','S1':'dashed','S2':'dotted'}[r['baseline']]}
    datasets['core'] = [clean(r) for r in core]
    datasets['qps'] = [clean(get(b,p)) for p in ('prefix_2k_q025','prefix_2k_q05','prefix_2k_q1') for b in ('S0','S1','S2')]
    datasets['context'] = [clean(get(b,p)) for p in ('prefix_1k_q1','prefix_2k_q1','prefix_4k_q1') for b in ('S0','S1','S2')]
    datasets['transfer'] = transfers
    datasets['cdf'] = []
    for b in ('S0','S1','S2'):
        times = sorted(r['ttft_s']*1000 for r in raw[b,'pressure_4k_q1'])
        datasets['cdf'] += [{'baseline':b,'ttft_ms':t,'cdf':(i+1)/len(times),'n':len(times),
                             'line_style':{'S0':'solid','S1':'dashed','S2':'dotted'}[b]} for i,t in enumerate(times)]
    sql_data, sql_text = query_raw(root,LABELS,rows)
    datasets['core']=[r for r in sql_data['metrics'] if r['point'] in ORDER[:3]]
    datasets['qps']=[r for p in ('prefix_2k_q025','prefix_2k_q05','prefix_2k_q1') for r in sql_data['metrics'] if r['point']==p]
    datasets['context']=[r for p in ('prefix_1k_q1','prefix_2k_q1','prefix_4k_q1') for r in sql_data['metrics'] if r['point']==p]
    datasets['cdf']=sql_data['cdf']
    datasets['transfer']=sql_data['transfer']
    sources = [
        {'id':id,'label':label,'path':f'report/queries/{name}.sql',
         'query':{'sql':sql_text[name], 'engine':'sqlite', 'language':'sql',
                  'tables_used':tables, 'description':'由scripts/v100_chart_queries.py将raw中原始JSON记录载入同名内存表，再执行本SQL。指标已与Python原始分析器逐项核对。'}}
        for id,name,label,tables in [('results','metrics','原始请求的SQLite独立聚合',['requests','itls','points']),
                                    ('requests','cdf','压力点原始请求的经验CDF',['requests']),
                                    ('transfer','transfer','原始拷贝事件的双向统计',['transfers','points'])]] + [
        {'id':'protocol','label':'已确认复现要求与实际方法','path':'docs/需求对照与指标字典.md'},
    ]
    charts = []
    def chart(id,title,dataset,x,y,ylabel,kind='line',color='baseline',source='results'):
        c = {'id':id,'title':title,'type':kind,'dataset':dataset,'sourceId':source,
             'subtitle':'2026年9月10日；seed42；每组100请求，压力点128请求；三张同规格V100',
             'showDescription':True,'layout':'full','valueFormat':'number',
             'encodings':{'x':{'field':x,'type':'nominal' if x=='label' else 'quantitative','label':{'qps':'计划QPS','actual_isl_mean':'实际平均输入tokens','label':'测点','ttft_ms':'TTFT(ms)'}[x]},
                          'y':{'field':y,'type':'quantitative','label':ylabel},
                          'color':{'field':color,'type':'nominal'}},
             'palette':{'kind':'categorical'},'legend':{'position':'bottom'},
             'settings':{'groupMode':'grouped','showPoints':'always','sort':'none'},
             'intent':'comparison' if kind=='bar' else 'custom',
             'question':title,'rationale':'按照复现要求对实际离散配置或经验分布做对照，不进行插值估计、统计外推或跨测点平均。'}
        if kind=='line':
            c['encodings']['lineStyle']={'field':'line_style','type':'nominal'}
        charts.append(c)
    chart('cache','核心负载的总缓存复用比例','core','label','cache_hit_ratio','复用tokens/输入tokens','bar')
    charts[-1]['valueFormat']='percent'
    chart('throughput','核心负载的输出吞吐','core','label','output_token_throughput','tokens/s','bar')
    for prefix,dataset,x in [('qps','qps','qps'),('context','context','actual_isl_mean')]:
        for m in ('ttft','itl'):
            chart(f'{prefix}_{m}',f"{'QPS' if prefix=='qps' else '输入长度'}与{m.upper()}中位数",dataset,x,f'{m}_ms_p50','ms')
    chart('cdf','压力4K的TTFT经验CDF','cdf','ttft_ms','cdf','累计比例',source='requests')
    charts[-1]['valueFormat']='percent'
    charts[-1]['subtitle']='每组128请求；QPS1；完整TTFT样本；连线仅指示经验累计分布'
    chart('transfer_bytes','S2各测点双向传输量','transfer','label','GB','GB','bar','direction','transfer')
    chart('transfer_bw','S2各测点有效传输带宽','transfer','label','GBps','GB/s','bar','direction','transfer')
    blocks = []
    for i,section in enumerate(re.split(r'(?=^## )',template,flags=re.M)):
        for j,part in enumerate(re.split(r'(\{\{chart:[a-z_]+\}\})',section)):
            if not part.strip(): continue
            if part.startswith('{{chart:'):
                blocks.append({'id':f'b{i}_{j}','type':'chart','chartId':part[8:-2],'layout':'full'})
            else:
                block={'id':f'b{i}_{j}','type':'markdown','body':part.strip()}
                blocks.append(block)
    title='实验三：Benchmark 1 + SGLang 复现报告'
    now=datetime.now(timezone.utc).isoformat()
    artifact={'surface':'report','manifest':{'version':1,'surface':'report','title':title,
              'generatedAt':now,'sources':sources,'blocks':blocks,'charts':charts},
              'snapshot':{'version':1,'status':'ready','generatedAt':now,'datasets':datasets},'sources':sources}
    out=root/'report'
    out.mkdir(exist_ok=True)
    (out/'artifact.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n')
    static = {'cache':'core_cache_throughput','throughput':'core_cache_throughput',
              'qps_ttft':'qps_latency','qps_itl':'qps_latency','context_ttft':'context_latency',
              'context_itl':'context_latency','cdf':'pressure_latency_cdf',
              'transfer_bytes':'offload_details','transfer_bw':'offload_details'}
    shown=set()
    for key,name in static.items():
        replacement = f'![{name}](../results/figures/{name}.png)' if name not in shown else ''
        shown.add(name)
        template=template.replace('{{chart:'+key+'}}',replacement)
    (out/'实验报告.md').write_text(template)
    with (root/'results/offload_details.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(transfers[0])); w.writeheader(); w.writerows(transfers)
    extra_figure(transfers,root/'results/figures')
    print(json.dumps({'report':str(out),'charts':len(charts),'blocks':len(blocks),'datasets':len(datasets)},ensure_ascii=False))


def extra_figure(transfers, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    fonts={f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams['font.family']=next((f for f in ('PingFang SC','Heiti TC','Noto Sans CJK SC','Arial Unicode MS') if f in fonts),'DejaVu Sans')
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
    for ax,key,title,unit in zip(axes,['GB','GBps'],['双向KV传输量','阻塞调用有效带宽'],['GB','GB/s']):
        for i,(direction,color,hatch) in enumerate([('GPU→CPU','#2774AE','//'),('CPU→GPU','#D27B2A','..')]):
            selected=[r for r in transfers if r['direction']==direction]
            ax.bar([x+(i-.5)*.36 for x in range(len(selected))],[float('nan') if r[key] is None else r[key] for r in selected],
                   width=.34,color=color,hatch=hatch,edgecolor='#333333',linewidth=.5,label=direction)
        ax.set_xticks(range(len(ORDER)),[LABELS[p].replace('/','\n') for p in ORDER],rotation=25,ha='right')
        ax.set(ylabel=unit,title=title,ylim=(0,None)); ax.grid(axis='y',color='#e8e8e8'); ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False); ax.legend(frameon=False)
    fig.suptitle('HiCache双向传输：7个测点，每点100请求，压力点128请求\nS2；seed42；GB为十进制；无回载的带宽留空；有效带宽包含阻塞等待',fontsize=12)
    output.mkdir(exist_ok=True)
    fig.savefig(output/'offload_details.png',dpi=180); plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('root',type=Path)
    make_report(parser.parse_args().root.resolve())
