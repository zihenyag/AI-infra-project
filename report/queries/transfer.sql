WITH directions AS (SELECT 'gpu_to_cpu' direction UNION ALL SELECT 'cpu_to_gpu')
SELECT p.point,p.label,CASE d.direction WHEN 'gpu_to_cpu' THEN 'GPU→CPU' ELSE 'CPU→GPU' END direction,
 COALESCE(SUM(t.bytes),0) bytes, COALESCE(SUM(t.bytes),0)/1e9 GB,
 COALESCE(SUM(t.time_s),0) time_s,
 SUM(t.bytes)/NULLIF(SUM(t.time_s),0)/1e9 GBps,
 COUNT(t.bytes) operations, p.n
FROM points p CROSS JOIN directions d
LEFT JOIN transfers t ON t.point=p.point AND t.direction=d.direction
WHERE p.baseline='S2' GROUP BY p.point,d.direction ORDER BY p.ordering,d.direction DESC;
