"""Standalone scientific SVG charts from freshly re-audited evidence.

Uses ReportLab's chart renderer. Missing cells are annotations, never zeros.
"""
import json
from pathlib import Path
from .regression import write_json,sha256
from .execution_plan import runtime_source_manifest


def data(request,root):
    if request['kind']=='paper_comparison':
        from .paper_comparison import report
        r=report(request['paper'],root,request['directories'],request['references']);points=[];ratios=[]
        for g in r['groups']:
            for v,s in g['variants'].items():points.append(dict(label=g['family']+' / '+v,value=s['mean'] if g['complete'] else None,sd=s['sample_stddev'] if g['complete'] else None,n=s['n'],complete=g['complete']))
            for v,s in (g['speedups'] or {}).items():ratios.append(dict(label=g['family']+' / '+v,value=s['summary']['mean'] if s['complete'] else None,sd=s['summary']['sample_stddev'] if s['complete'] else None,n=s['summary']['n'],complete=s['complete']))
        return r,[dict(name='qualified-rate',title='Deadline-qualified action rate Q',unit='Hz',points=points),dict(name='paired-speedup',title='Paired ROSA / baseline',unit='ratio',points=ratios)],'GPU' if r['gpu_results'] else 'CPU / MIXED / NO VERIFIED GPU DATA'
    if request['kind']=='pipeline_comparison':
        from .comparison_report import report
        r=report(request['manifest'],root);points=[dict(label=g['condition_id']+' / '+g['variant'],value=g['observed_weighted_Q']['mean'] if g['complete'] else None,sd=g['observed_weighted_Q']['sample_stddev'] if g['complete'] else None,n=g['observed_weighted_Q']['n'],complete=g['complete']) for g in r['groups']]
        gpu=all(v['execution_mode']=='gpu' for v in r['conditions'].values())
        return r,[dict(name='qualified-rate',title='Deadline-qualified action rate Q',unit='Hz',points=points)],'GPU' if gpu else 'CPU / MIXED DATA — not GPU performance'
    raise ValueError('Unknown audited figure input')


def bar_svg(series,label,path):
    from reportlab.graphics.shapes import Drawing,String,Line
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics import renderSVG
    from reportlab.lib.colors import HexColor
    points=series['points'];width=max(860,len(points)*135+100);height=560;d=Drawing(width,height)
    d.add(String(35,530,series['title'],fontName='Helvetica-Bold',fontSize=18))
    d.add(String(35,506,label,fontSize=11,fillColor=HexColor('#a04116')))
    d.add(String(35,485,'Error bars: sample SD across declared repetitions; not confidence intervals.',fontSize=10))
    d.add(String(35,465,'Incomplete families suppressed; missing values are not zero. No paper-equivalence claim.',fontSize=10))
    if not points:
        d.add(String(40,300,'No complete ratio series available.',fontSize=14));renderSVG.drawToFile(d,str(path));return
    chart=VerticalBarChart();chart.x=65;chart.y=160;chart.width=width-100;chart.height=270
    chart.data=[[p['value'] or 0 for p in points]];chart.categoryAxis.categoryNames=[p['label']+f" (n={p['n']})" for p in points]
    chart.categoryAxis.labels.angle=35;chart.categoryAxis.labels.boxAnchor='ne';chart.categoryAxis.labels.fontSize=8
    chart.valueAxis.valueMin=min([0.]+[(p['value'] or 0)-(p['sd'] or 0) for p in points]);chart.valueAxis.valueMax=max([1.]+[(p['value'] or 0)+(p['sd'] or 0) for p in points])*1.15
    chart.valueAxis.labelTextFormat='%.3g';chart.bars[0].fillColor=HexColor('#237b93');d.add(chart)
    d.add(String(5,445,series['unit'],fontSize=11))
    slot=chart.width/len(points)
    for i,p in enumerate(points):
        x=chart.x+(i+.5)*slot
        if p['value'] is None:d.add(String(x,175,'MISSING',textAnchor='middle',fontSize=9,fillColor=HexColor('#a04116')))
        elif p['sd'] is not None:
            low=chart.y+chart.height*(p['value']-p['sd']-chart.valueAxis.valueMin)/(chart.valueAxis.valueMax-chart.valueAxis.valueMin);high=chart.y+chart.height*(p['value']+p['sd']-chart.valueAxis.valueMin)/(chart.valueAxis.valueMax-chart.valueAxis.valueMin)
            for a,b,c,e in ((x,low,x,high),(x-4,low,x+4,low),(x-4,high,x+4,high)):d.add(Line(a,b,c,e,strokeColor=HexColor('#182b35')))
    renderSVG.drawToFile(d,str(path))


def export(request,root,output):
    r,series,label=data(request,root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    write_json(out/'request.json',request);write_json(out/'audited-data.json',r);write_json(out/'plot-data.json',series)
    for s in series:bar_svg(s,label,out/(s['name']+'.svg'))
    write_json(out/'manifest.json',dict(source_manifest=runtime_source_manifest(),label=label,paper_equivalence_claim=False,
        files={p.name:sha256(p) for p in out.iterdir() if p.is_file()}))
    return dict(status='complete',figures=[str(out/(s['name']+'.svg')) for s in series],input_status=r['status'])
