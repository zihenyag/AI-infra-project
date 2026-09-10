"""用SQLite直接聚合原始请求/传输事件，独立核对HTML图表；不更改原始记录。"""
import json
from pathlib import Path
import sqlite3


METRICS_SQL = '''WITH ranked AS (
 SELECT baseline, point, ttft_ms,
 ROW_NUMBER() OVER(PARTITION BY baseline, point ORDER BY ttft_ms) rn,
 COUNT(*) OVER(PARTITION BY baseline, point) n FROM requests
), ttft AS (
 SELECT baseline, point, AVG(ttft_ms) ttft_ms_p50 FROM ranked
 WHERE rn IN ((n+1)/2,(n+2)/2) GROUP BY baseline, point
), itl_ranked AS (
 SELECT baseline, point, value_ms,
 ROW_NUMBER() OVER(PARTITION BY baseline, point ORDER BY value_ms) rn,
 COUNT(*) OVER(PARTITION BY baseline, point) n FROM itls
), itl AS (
 SELECT baseline, point, AVG(value_ms) itl_ms_p50 FROM itl_ranked
 WHERE rn IN ((n+1)/2,(n+2)/2) GROUP BY baseline, point
), totals AS (
 SELECT baseline, point, COUNT(*) completed, SUM(prompt_tokens) prompt_tokens,
 SUM(output_tokens) output_tokens, SUM(cached_tokens) cached_tokens,
 AVG(prompt_tokens) actual_isl_mean, AVG(client_wait_s) client_wait_s_mean
 FROM requests GROUP BY baseline, point
)
SELECT totals.*, ttft.ttft_ms_p50, itl.itl_ms_p50, points.qps, points.duration_s,
 points.label, 1.0*totals.cached_tokens/totals.prompt_tokens cache_hit_ratio,
 1.0*totals.output_tokens/points.duration_s output_token_throughput,
 CASE totals.baseline WHEN 'S0' THEN 'solid' WHEN 'S1' THEN 'dashed' ELSE 'dotted' END line_style
FROM totals JOIN ttft USING(baseline,point) JOIN itl USING(baseline,point)
JOIN points USING(baseline,point)
ORDER BY points.ordering,totals.baseline'''

CDF_SQL = '''SELECT baseline, ttft_ms,
 1.0*ROW_NUMBER() OVER(PARTITION BY baseline ORDER BY ttft_ms)/COUNT(*) OVER(PARTITION BY baseline) cdf,
 COUNT(*) OVER(PARTITION BY baseline) n,
 CASE baseline WHEN 'S0' THEN 'solid' WHEN 'S1' THEN 'dashed' ELSE 'dotted' END line_style
FROM requests WHERE point='pressure_4k_q1' ORDER BY baseline,ttft_ms'''

TRANSFER_SQL = '''WITH directions AS (SELECT 'gpu_to_cpu' direction UNION ALL SELECT 'cpu_to_gpu')
SELECT p.point,p.label,CASE d.direction WHEN 'gpu_to_cpu' THEN 'GPU→CPU' ELSE 'CPU→GPU' END direction,
 COALESCE(SUM(t.bytes),0) bytes, COALESCE(SUM(t.bytes),0)/1e9 GB,
 COALESCE(SUM(t.time_s),0) time_s,
 SUM(t.bytes)/NULLIF(SUM(t.time_s),0)/1e9 GBps,
 COUNT(t.bytes) operations, p.n
FROM points p CROSS JOIN directions d
LEFT JOIN transfers t ON t.point=p.point AND t.direction=d.direction
WHERE p.baseline='S2' GROUP BY p.point,d.direction ORDER BY p.ordering,d.direction DESC'''


def query_raw(root, labels, expected):
    db=sqlite3.connect(':memory:'); db.row_factory=sqlite3.Row
    db.executescript('''CREATE TABLE points(baseline TEXT,point TEXT,qps REAL,duration_s REAL,label TEXT,ordering INTEGER,n INTEGER);
    CREATE TABLE requests(baseline TEXT,point TEXT,ttft_ms REAL,prompt_tokens INTEGER,output_tokens INTEGER,cached_tokens INTEGER,client_wait_s REAL);
    CREATE TABLE itls(baseline TEXT,point TEXT,value_ms REAL);
    CREATE TABLE transfers(point TEXT,direction TEXT,bytes INTEGER,time_s REAL);''')
    for row in expected:
        folder=root/'raw'/row['source']
        b,p=row['baseline'],row['point']
        result=json.loads((folder/'result.json').read_text())
        db.execute('INSERT INTO points VALUES (?,?,?,?,?,?,?)',(b,p,result['point']['qps'],result['measurement']['duration_s'],labels[p],list(labels).index(p),result['point']['n']))
        for line in (folder/'requests.jsonl').read_text().splitlines():
            r=json.loads(line); m=r['meta_info']
            db.execute('INSERT INTO requests VALUES (?,?,?,?,?,?,?)',(b,p,r['ttft_s']*1000,m['prompt_tokens'],m['completion_tokens'],m['cached_tokens'],r['client_wait_s']))
            db.executemany('INSERT INTO itls VALUES (?,?,?)',[(b,p,v*1000) for v in r['itls_s']])
        if b=='S2':
            for e in json.loads((folder/'events.json').read_text()):
                if e['kind']=='transfer':
                    db.execute('INSERT INTO transfers VALUES (?,?,?,?)',(p,e['direction'],e['bytes'],e['host_call_s']))
    query_dir=root/'report/queries'; query_dir.mkdir(parents=True,exist_ok=True)
    queries={'metrics':METRICS_SQL,'cdf':CDF_SQL,'transfer':TRANSFER_SQL}
    output={}
    for name,sql in queries.items():
        output[name]=[dict(r) for r in db.execute(sql)]
        (query_dir/f'{name}.sql').write_text(sql+';\n')
    lookup={(r['baseline'],r['point']):r for r in expected}
    for row in output['metrics']:
        ref=lookup[row['baseline'],row['point']]
        for key in ('ttft_ms_p50','itl_ms_p50','cache_hit_ratio','output_token_throughput','prompt_tokens','output_tokens','cached_tokens','actual_isl_mean'):
            assert abs(row[key]-ref[key]) <= 1e-9*max(1,abs(ref[key])), (row['point'],key)
    for row in output['transfer']:
        ref=lookup['S2',row['point']]
        d='gpu_to_cpu' if row['direction']=='GPU→CPU' else 'cpu_to_gpu'
        assert row['bytes']==ref[d+'_bytes'] and row['operations']==ref[d+'_operations']
        assert abs(row['time_s']-ref[d+'_host_call_s'])<1e-8
    db.close()
    return output,queries
