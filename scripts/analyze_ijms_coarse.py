"""Read returned coarse results and write a data-grounded fine-screen proposal.

Does not run simulations or modify the input results. Run from the repository:
  .venv/Scripts/python.exe -B scripts/analyze_ijms_coarse.py
"""
import argparse
import csv
import gzip
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from spine_sim.research_protocols import EstablishmentProtocol, evaluate_establishment


COMPLETE = 'BALANCED_COMPLETE'
MATERIALS = ['P240', 'P100', 'P40', 'fired_brick_standard', 'rough_wall']
LABELS = dict(zip(MATERIALS, ['P240', 'P100', 'P40', '红砖', '混凝土']))


def fine_designs():
    result = {}

    def add(group, radius, angles, stiffnesses, layouts, pitches=(6,)):
        for angle in angles:
            for stiffness in stiffnesses:
                for layout in layouts:
                    for pitch in pitches:
                        name = f'r{radius}_{angle}_{stiffness}_{layout}_p{pitch}'
                        result.setdefault(name, group)

    add('主细化', 100, ['a70', 'a75', 'a80'],
        ['spring1000', 'spring2000', 'spring3500', 'spring5000'], ['2x2', '2x5'])
    add('小针尖对照', 50, ['a70', 'a80'], ['spring2000', 'spring5000'], ['2x2', '2x5'])
    add('布局对照', 100, ['a70', 'a80'], ['spring2000', 'spring5000'], ['3x3', '5x2', '4x4'])
    add('间距对照', 100, ['a70', 'a80'], ['spring2000', 'spring5000'], ['2x5'], [4, 5])
    for radius in [50, 100]:
        add('固定安装对照', radius, ['a70', 'a80'], ['fixed0'], ['2x2'])
    add('梯度对照', 100, ['gradient60to80'], ['spring2000', 'spring5000'], ['2x2', '2x5'])
    add('软弹簧边界', 100, ['a80'], ['spring100', 'spring400'], ['2x5'])
    add('粗筛高完成率补充', 100, ['a80'], ['spring5000'], ['5x3'], [4])
    add('粗筛高完成率补充', 100, ['a80'], ['spring5000'], ['3x3'], [5])
    return result


def mechanism_designs():
    """Paired controls for the paper, all already present in the coarse design."""
    result={}
    def add(group,radius,angle,mount,layout='4x4',pitch=5):
        result.setdefault(f'r{radius}_{angle}_{mount}_{layout}_p{pitch}',group)
    for radius in [50,100]:
        for mount in ['spring100','spring400','spring800','spring1000','spring2000','spring5000','fixed0']:
            add('柔顺与半径',radius,'a70',mount)
    for angle in ['a60','a80','gradient60to80']:
        for mount in ['spring800','spring5000']:
            add('角度与梯度',100,angle,mount)
    for mount in ['spring800','spring5000']:
        for layout in ['2x2','3x3','5x5']:
            add('固定间距规模',100,'a70',mount,layout)
        for layout in ['2x5','5x2']:
            add('同针数转向',100,'a70',mount,layout)
        for pitch in [4,6]:
            add('间距',100,'a70',mount,'2x5',pitch)
    return result


def median(values):
    values = [v for v in values if np.isfinite(v)]
    return float(np.median(values)) if values else float('nan')


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, r)) + ' |' for r in rows]])


def establishment_from_trace(trace, preload):
    """Preserve interval validity: a rearrangement flag invalidates its incoming edge."""
    preload_rows = [r for r in trace['rows'] if r['phase'] == 'preload']
    path = preload_rows[-1:] + [r for r in trace['rows'] if r['phase'] == 'drag']
    x, force = [], []
    for a, b in zip(path[:-1], path[1:]):
        if b['x_m'] <= a['x_m']:
            continue
        valid = not b.get('unresolved_rearrangement', b.get('reconfigured', False))
        x.extend([a['x_m'], b['x_m']])
        force.extend([a['T_N'], b['T_N']] if valid else [float('nan'), float('nan')])
    return evaluate_establishment(
        x, force, accepted=np.ones(len(x), dtype=bool), valid=np.isfinite(force),
        protocol=EstablishmentProtocol(preload, .0005, (0, .01)),
    ).established


def write_prototype_selection(rows,root,output):
    """Separate physical specimens from the still-undecided fine-screen size."""
    designs={'A':'r100_a70_spring5000_2x2_p6',
             'B':'r50_a70_spring5000_2x2_p6',
             'C':'r100_a70_spring5000_4x4_p6'}
    lookup={(r['design'],r['material'],r['preload_N'],int(r['sample_index'])):r for r in rows}
    cache={}
    def load(label,material,preload,sample):
        key=(label,material,preload,sample)
        if key not in cache:
            row=lookup[designs[label],material,preload,sample]
            with gzip.open(root/row['trace_file'],'rt',encoding='utf-8') as f:
                trace=json.load(f)
            path=[r for r in trace['rows'] if r['phase']=='preload'][-1:]+[r for r in trace['rows'] if r['phase']=='drag']
            cache[key]=(row,trace,path)
        return cache[key]
    def paired(first,p1,second,p2,material):
        records=[]
        for sample in range(4):
            r1,j1,a=load(first,material,p1,sample);r2,j2,b=load(second,material,p2,sample)
            if not (r1['ok'] and r2['ok']) or j1['selected_start_xy_m']!=j2['selected_start_xy_m']:
                continue
            x1=np.array([r['x_m'] for r in a]);x2=np.array([r['x_m'] for r in b])
            t1=np.array([r['T_N'] for r in a]);t2=np.array([r['T_N'] for r in b])
            x=np.unique(np.r_[x1,x2]);mid=(x[:-1]+x[1:])/2;dx=np.diff(x)
            i1=np.searchsorted(x1,mid)-1;i2=np.searchsorted(x2,mid)-1
            valid=(i1>=0)&(i1<len(x1)-1)&(i2>=0)&(i2<len(x2)-1)
            i1=np.clip(i1,0,len(x1)-2);i2=np.clip(i2,0,len(x2)-2)
            valid &= np.array([not a[i+1].get('unresolved_rearrangement',a[i+1].get('reconfigured',False)) for i in i1])
            valid &= np.array([not b[i+1].get('unresolved_rearrangement',b[i+1].get('reconfigured',False)) for i in i2])
            length=dx[valid].sum()
            means=[float(np.dot(np.interp(mid[valid],xx,tt),dx[valid])/length) for xx,tt in [(x1,t1),(x2,t2)]]
            records.append((sample,*means,length*1000))
        return records
    lines=['# IJMS 实体样件候选与细筛规模说明',
        '> 最新决定：统一r100针型、206种细筛构型，见[细筛构型定稿](D:/Code/Spine_Sim/docs/IJMS细筛构型定稿.md)。下文A/B/C是此前的历史对照，r50与r100双针型制造建议不再执行。',
        '2026-09-13。仅用已返回粗筛数据选型，未运行新仿真，也未冻结细筛数量。',
        '**研究目标修正：预载与阵列规模、柔顺和表面相匹配，以获得更大的有效承载和合适的建立距离；'
        '不能将其概括成一味减小预载。** 上一版仿真中关于10×10、20×20规模匹配的观察由用户提供，'
        '本次没有重读那批旧数据，也不把它混入当前数值。',
        '## 粗筛到底有多少种设计',
        '2520种结构＝2种半径×5种角度方案×7种安装方案×12种布局×3种间距。'
        '每种结构有3预载×5材料×4地形＝60个工况，共151200个工况。'
        '这里的结构设计数、加载工况数、实体制造件数是不同概念。',
        table(['轴','粗筛内容'],[
            ('针尖半径','50、100 μm'),('角度方案','50、60、70、80°及60→80°梯度'),
            ('安装方案','100、400、800、1000、2000、5000 N/m弹簧，以及固定安装'),
            ('布局','2×2、3×3、4×4、5×5、2×5、5×2、2×8、8×2、3×5、5×3、3×8、8×3'),
            ('间距','4、5、6 mm'),('每设计加载与样本','P=0.5/1/2 N；5材料；4个地形实现')]),
        '细筛数量由用户接下来确定，之前64/95设计只是未执行的建议。'
        '如果仍按每设计60个条件计算，100/200/400/800设计分别对应6000/12000/24000/48000工况。'
        '这只是规模换算，不是新增计算清单；相同设置的旧结果不重复计算。',
        '## 建议先考虑的三个实体构型',
        table(['样件','布局/针数','半径','安装角','每针弹簧','间距','作用'],[
            ('A','2×2 / 4','100 μm','70°','5000 N/m','6 mm','小阵列基准'),
            ('B','2×2 / 4','50 μm','70°','5000 N/m','6 mm','只改变针尖，验证粗糙材料上阻力差异'),
            ('C','4×4 / 16','100 μm','70°','5000 N/m','6 mm','只改变规模，验证规模与预载匹配')]),
        '共同沿用粗筛几何：主体直径1 mm、渐细段2 mm、均匀角露出4 mm、名义弹簧行程4 mm。'
        '5000 N/m＝5 N/mm。针尖中心跨度为A/B的6×6 mm、C的18×18 mm；这不是背板外形尺寸。'
        '以上是候选参数，不是最终夹具加工图。A/B可考虑同一背板换针，C采用相同导向/弹簧结构，'
        '便于让样件差异对应模型变量。',
        table(['样件','粗筛设计ID','到达10 mm'],[
            (label,design,f'{sum(r["ok"] for r in rows if r["design"]==design)}/60') for label,design in designs.items()]),
        '## 现有数据支持什么差异',
        '以下均为原始轨迹后处理：同材料、同地形编号、同实际起点，只在双方共同已解析的区间积分。'
        '每项报告四个地形的中位数，不按路径站点增加样本数。'
        'T为恒总预载拖动中的阻力；这里不把它当作另一个尚未定义的静态极限承载试验。']
    comparison_rows=[]
    comparisons=[('针尖差异','A',1.,'B',1.,'rough_wall'),
                 ('针尖差异','A',1.,'B',1.,'fired_brick_standard'),
                 ('同总P下的规模对照','A',1.,'C',1.,'fired_brick_standard'),
                 ('规模—预载匹配','A',.5,'C',2.,'P100'),
                 ('规模—预载匹配','A',.5,'C',2.,'P40'),
                 ('规模—预载匹配','A',.5,'C',2.,'fired_brick_standard')]
    detailed=[]
    for role,a,p1,b,p2,m in comparisons:
        records=paired(a,p1,b,p2,m)
        comparison_rows.append((role,LABELS[m],f'{a}: P={p1:g} N',f'{median([r[1] for r in records]):.3f}',
            f'{b}: P={p2:g} N',f'{median([r[2] for r in records]):.3f}',
            f'{sum(r[2]>r[1] for r in records)}/{len(records)}',f'{min(r[3] for r in records):.3f}–{max(r[3] for r in records):.3f}'))
        for s,t1,t2,length in records:detailed.append((a,p1,b,p2,LABELS[m],s,f'{t1:.6f}',f'{t2:.6f}',f'{length:.6f}'))
    lines += [table(['比较','材料','前者','T中位数/N','后者','T中位数/N','后者更高的地形数','共同已解析长度/mm'],comparison_rows),
        'A→C的匹配组将针数从4增加至16，总预载从0.5增加至2 N，名义P/N均为0.125 N/针。'
        '实际每针分载仍由力学平衡求出，不强制均分。三种列出的材料在四个地形上均出现更大绝对阻力，'
        '同时保留同总P对照，才能区分增加规模和增加预载的共同作用；不能把匹配组当成纯针数效应。',
        'B在混凝土P=1 N的四个配对上平均阻力均高于A，适合验证针尖效应。'
        '但其低分位阻力没有同步改善，且P40有明显尖峰；因此B的角色是“较高平均阻力与波动的对照”，'
        '不是已经证明所有持续承载指标更优。实体应同时记录平均力、低分位力、力—位移及建立距离。',
        'C在同P=1 N的红砖对照中，四个地形的平均阻力均低于A；当按上述名义P/N匹配提高P后，'
        '其绝对阻力显著提高。这组对照符合本项目要研究的规模—预载关系，并不支持“大阵列天然更差”。',
        '## 与实验和后续细筛的衔接',
        '建议三个样件都保留0.5/1/2 N共同预载档，再重点比较A@0.5 N与C@2 N。'
        'B的形貌差异优先看混凝土/红砖；A—C规模匹配优先使用P100、P40、红砖。'
        'P240可保留作区分度较低的参照，不能为了获得明显差异只挑一块有利表面。',
        '实物使用六维力传感器和直线模组，核对实际法向反力；Y自由或受限须与模型对应。'
        '同型号材料的新样片与仿真合成地形并非同一表面，当前四个实现提供的是候选趋势，不是实验差值保证。'
        '导向、露出与限位细节按真实装配确定，尤其4 mm完全缩回不能自动当成仍有露出长度的承载限位。',
        '细筛可以保留数百种设计，与先制造三个代表构型并不冲突。'
        '10×10、20×20未包含在当前粗筛中，不能直接给出它们的推荐预载。'
        '本次P/N匹配只是跨规模比较的一条明确路径，不能线性外推为大阵列最优预载定律。',
        '## 配对数值明细',
        table(['前者','P/N','后者','P/N','材料','地形序号','前者T/N','后者T/N','共同长度/mm'],detailed)]
    output.write_text('\n\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('E:/TestData/IJMS/results'))
    parser.add_argument('--report', type=Path, default=Path('docs/IJMS粗筛分析与细筛规划.md'))
    args = parser.parse_args()
    with (args.results / 'summary.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        parts = r['design'].split('_')
        r.update(zip(['radius', 'angle', 'mount', 'layout', 'pitch'], parts))
        for k in ['preload_N', 'solve_s', 'coverage_mm', 'resolved_distance_mm', 'mean_T_resolved_N']:
            r[k] = float(r[k]) if r[k] else float('nan')
        r['ok'] = r['status'] == COMPLETE
        r['whole'] = r['ok'] and r['resolved_distance_mm'] >= 10 - 1e-8
        r['ratio'] = r['mean_T_resolved_N'] / r['preload_N']
    keys = {(r['design'], r['material'], r['preload_N'], r['sample_index']): r for r in rows}
    counts = Counter(r['status'] for r in rows)
    with sqlite3.connect(args.results.joinpath('case_index.sqlite3').resolve().as_uri() + '?mode=ro', uri=True) as db:
        dbcounts = dict(db.execute('SELECT status, count(*) FROM cases GROUP BY status'))
    files = {p.relative_to(args.results).as_posix() for p in args.results.glob('s*/*/*.json.gz')}
    missing = [r for r in rows if r['trace_file'] not in files]
    bydesign = defaultdict(list)
    for r in rows:
        bydesign[r['design']].append(r)
    sections = []
    put = sections.append
    put('# IJMS 粗筛分析与下一轮细筛规划\n\n日期：2026-09-13。状态：分析与规划，未启动仿真。')
    put('> 最新细筛以[统一针型206构型定稿](D:/Code/Spine_Sim/docs/IJMS细筛构型定稿.md)为准。下文保留粗筛数据分析及历史方案；旧64/95构型、34构型补算和双针型制造建议不再作为当前执行范围。')
    put('用户最新决定：不重跑或补齐粗筛，先仅凭现有数据筛选设计。'
        '此前34设计补算及全库续算建议取消；34设计仅作为旧数据机制对照集，64设计仅保留为后续参考，不启动计算。'
        '论文目标是使预载与阵列规模、柔顺和表面相匹配，提高有效承载并获得合适的建立距离；不追求一味减小预载。'
        '两种半径、软—中—硬弹簧和固定安装对照都保留；固定安装异常高值暂不作为优胜依据。')
    put('## 当前离线筛选结论')
    excluded=[r for r in rows if r['mount']=='spring100' and r['layout']=='2x2'
              and r['angle'] in ['a50','a60','a70','a80'] and r['preload_N']==2]
    excluded_counts=Counter(r['status'] for r in excluded)
    put('**可以明确排除的是一组设计—载荷组合，不是整档参数在所有任务下都无效。** '
        '若候选必须覆盖总预载2 N，则剔除2×2、100 N/m、均匀角50/60/70/80°、'
        'r50/r100、间距4/5/6 mm这24个设计的高预载用途。'
        f'相应{len(excluded)}个已有工况无一完成10 mm：'+
        '、'.join(f'{s} {n}' for s,n in excluded_counts.most_common())+'。')
    put('排除依据不只是完成率。当前刚性针、无构形力、均匀安装角的弹簧分支满足'
        '`T cosθ + P sinθ = k Σs_i`。每针露出4 mm，达到4 mm时完全缩回，不能继续作为硬限位承载；'
        '因此4针×100 N/m给出`k Σs_i < 1.6 N`。当P=2 N，'
        '`T < (1.6−2 sinθ)/cosθ`。这是当前配置的必要上界，不是可达到的容量预测，'
        '不外推到梯度角、固定安装或其他物理模型。')
    put(table(['均匀角','P=2 N时T上界（N）','设计用途判断'],[
        (a,f'{(1.6-2*math.sin(math.radians(a)))/math.cos(math.radians(a)):.3f}',
         '无法满足最低0.5 N持续阻力目标') for a in [50,60,70,80]]))
    put('这些设计在P=0.5/1 N下仍可能有研究价值，不从旧数据中删除；'
        '梯度角100 N/m涉及已修复的上硬限位，不能套用上述排除规则。')
    put(table(['处理','对象','理由和保留范围'],[
        ('剔除高预载用途','上述24个2×2/100 N/m均匀角设计','P=2 N下既有数据和轴向必要上界一致；低预载用途保留'),
        ('暂不进入性能优选','固定安装中已出现大转角/异常尖峰的工况','小转角模型适用性不足；保留机制对照，不据高值选赢家'),
        ('暂不扩大','124个没有任何完整10 mm记录的设计','证据不足，节约后续计算；不能把技术失败判成结构无效'),
        ('降优先级','100 N/m的大范围参数细化、5×5/3×8/8×3的大范围扩展','现有可比较路径少；保留配对机制参照，不据此宣称软或大阵列必然差'),
        ('继续保留','两种半径、50/60/70/80°、800/1000/2000/5000 N/m的材料相关候选','当前没有跨材料一致的劣势，不能整档删除')]))
    put('进一步限定双方均完整解析10 mm、同材料/预载/地形/实际起点的弹簧配对后，'
        '50°在P100、P240上的已解析平均阻力经常高于70°，因此撤回“可以只保留70–80°”的泛化理解。'
        '这仍是完成子集的条件比较，只支持保留小角度材料专项候选，不证明50°全局最优。')
    ptable=[]
    for first,second in [('a50','a70'),('a70','a80')]:
        values=defaultdict(list)
        for r in rows:
            if r['angle']!=first or not r['mount'].startswith('spring'):continue
            q=keys.get((r['design'].replace('_'+first+'_','_'+second+'_'),r['material'],r['preload_N'],r['sample_index']))
            if q and r['whole'] and q['whole'] and r['selected_start_xy_m']==q['selected_start_xy_m']:
                values[r['material']].append(q['ratio']-r['ratio'])
        for m in MATERIALS:
            if not values[m]:continue
            v=values[m]
            ptable.append((first+'→'+second,LABELS[m],len(v),f'{median(v):+.4f}',f'{100*np.mean(np.asarray(v)>0):.1f}%'))
    put(table(['角度比较','材料','可比较配对数','后者−前者的R差中位数','后者更高比例'],ptable))
    put('实际收缩方式：先使用第4节六个已有结果锚点作为性能候选，'
        '同时保留r50/a60/k2000/2×2/p6作为小角度材料专项对照（P100和P240各12/12到达终点；'
        '混凝土/红砖证据不足，不当跨材料通用候选）。'
        '仅对候选已有原始轨迹计算2/5/10 mm持续建立与分载，不重算缺失case。'
        '其余设计保留原始数据，以“排除某用途、暂缓、证据不足”分别标记。')
    put('## 与论文设计的对应\n\n依据：'
        '[研究与论文规划](D:/Pdrive/Document/论文/IJMS/01_研究与论文规划.md)、'
        '[仿真规划](D:/Pdrive/Document/论文/IJMS/02_仿真规划.md)、'
        '[实体实验规划](D:/Pdrive/Document/论文/IJMS/03_实体实验规划.md)，'
        '以及2026-09-12整合的[唯一机理母稿](D:/Pdrive/Document/论文/IJMS/理论依据/IJMS_机理推导母稿.md)。'
        '这些文件中的早期待定数值不覆盖后来已冻结的预载、针形和粗筛协议；本次没有修改PDrive原稿。')
    put(table(['论文问题','下一轮必须保留的证据'],[
        ('柔顺能否改善分载并缩短建立距离','同半径、角度、4×4/p5、地形和实际落点的100至5000 N/m及固定安装对照；不能按完成率删软弹簧'),
        ('规模与预载是否匹配','2×2、3×3、4×4、5×5在相同间距与总P下比较；用同一轨迹的2/5/10 mm窗口研究行程'),
        ('角度与排列如何作用','均匀角与长度补偿梯度配对；2×5与5×2同针数转向；保留间距对照'),
        ('表面是否改变结构排序','五材料分别报告；受控幅值/相关长度组和独立地形是后续待执行组，不以材料标签替代单因素证据'),
        ('怎样选实体构型','选短建立距离、分载改善、性能权衡及反例角色；不只制造平均力最大的几个构型')]))
    put('当前折中模型仍不能替代母稿中有限弯曲、构形力、完整几何/强度边界的全部验证。'
        '保留用户已授权的刚性针分流用于筛选；当论文结论直接涉及针杆弯曲或构形耦合时，'
        '需要相应模型的少量匹配对照，不能用当前近似结果声称这些机制已验证。'
        '本次不自动恢复原严格模型的大规模队列。')
    put('## 1. 返回数据是否齐全')
    put(f'输入目录：`{args.results.resolve()}`。汇总表有{len(rows):,}行、{len(keys):,}个唯一工况，'
        f'覆盖{len(bydesign)}设计×5材料×3预载×4地形实现。CSV与SQLite状态计数'
        f'{"完全一致" if dict(counts) == dbcounts else "不一致"}。'
        f'目录有{len(files):,}份压缩轨迹；{len(missing)}条记录没有对应轨迹，'
        f'其中执行错误{sum(r["status"] == "EXECUTION_ERROR" for r in missing)}条。'
        '这里核对了文件存在性，未逐一解压全库。')
    put(table(['终止状态', '数量', '占全部工况'],
              [(s, f'{n:,}', f'{100*n/len(rows):.2f}%') for s, n in counts.most_common()]))
    errors = Counter(r['error'] for r in rows if r['error'])
    put('执行错误具体为：' + '；'.join(f'`{e}`：{n}例' for e, n in errors.most_common()) + '。'
        '预算停止不能视为物理失败，轴向范围停止不能解释成材料断裂。')
    budget_rows=[r for r in rows if r['status']=='BALANCED_BUDGET_LIMIT']
    budget_drag=sum(bool(r['selected_placement_index']) for r in budget_rows)
    put(f'预算停止占所有未完成项的{100*len(budget_rows)/(len(rows)-counts[COMPLETE]):.2f}%。'
        f'按换点调度器的状态语义，{budget_drag:,}例已选中起点、建立预载后在拖动中耗尽预算，'
        f'{len(budget_rows)-budget_drag:,}例仍停在预载搜索。原45 s（固定60 s）由预载和整段拖动共用，'
        '不是每个阶段各有45/60 s；换点合计120 s也不是每个落点都能得到完整预算。'
        '复杂接触、软弹簧大回缩及反复缩步会用尽预算，机器变快或增加worker也不会取消这个CPU时间上限。')
    put(f'记录的模型为`ijms-balanced-condensed-contact-5`（{sum(bool(r["model"]) for r in rows):,}条；'
        '执行错误行缺少模型字段）。抽查原始结果确认服务器仍用25 μm路径步长、20 μm包络网格、'
        '10 μm平滑、1 μm摩擦过渡；弹簧安装用刚性针、固定安装用凝聚弹性针。'
        '未将旧纯趋势模型数据混入分析。模型字段一致不等于逐文件源码完全一致已验证。')
    whole = sum(r['whole'] for r in rows)
    gaps = [10-r['resolved_distance_mm'] for r in rows if r['ok']]
    put(f'完整到达10 mm有{counts[COMPLETE]:,}例，但其中{counts[COMPLETE]-whole:,}例仍含未解析重排区间。'
        f'全10 mm均已解析的仅{whole:,}例，占全库{100*whole/len(rows):.2f}%。'
        f'到达终点的轨迹中，缺口长度90%分位为{np.quantile(gaps,.9):.3f} mm，最大{max(gaps):.4f} mm。')
    put('本报告的阻力比R定义为`mean_T_resolved_N / preload_N`，只用于描述已解析区间；'
        '它不是完整10 mm的Jnet。现有汇总用有效区间的梯形积分除以有效长度，'
        'p90也按区间长度加权。缺失路径不能补零，不能用已解析长度替代正式窗口分母。'
        '即使标为BALANCED_COMPLETE，也不自动获得持续建立、全程稳定或模型有效的结论。')
    put('## 2. 材料与参数趋势')
    mrows = []
    for mat in MATERIALS:
        rs = [r for r in rows if r['material'] == mat]
        ok = [r for r in rs if r['ok']]
        mrows.append((LABELS[mat], f'{len(ok):,}/{len(rs):,}', f'{100*len(ok)/len(rs):.2f}%',
                      sum(r['whole'] for r in rs), f'{median([r["ratio"] for r in ok]):.3f}',
                      sum(r['status'] == 'BALANCED_BUDGET_LIMIT' for r in rs)))
    put(table(['材料', '到达10 mm', '完成率', '10 mm均解析', '完成子集R中位数', '预算停止'], mrows))
    put('完成子集随材料与设计变化，上表R不能当作全体设计的无偏性能比较。'
        'P240上的R集中在约0.42附近，区分度小；P100、P40适合观察几何差异，'
        '红砖和混凝土必须同时报告完成与缺失情况。保留全部五种材料，避免把难求解材料剔除后选赢家。')
    for field, title in [('radius', '针尖半径'), ('angle', '安装角'), ('mount', '安装与刚度'), ('preload_N', '总预载')]:
        group = []
        for value in sorted({r[field] for r in rows}, key=str):
            rs = [r for r in rows if r[field] == value]
            group.append((value, f'{100*sum(r["ok"] for r in rs)/len(rs):.2f}%',
                          f'{median([r["ratio"] for r in rs if r["ok"]]):.3f}'))
        put(f'### {title}\n\n' + table(['档位', '完成率', '完成子集R中位数'], group))
    put('### 配对比较\n\n下面控制其余设计参数、材料、预载和地形编号；阻力差只使用双方均到达10 mm、'
        '且记录的实际起点相同的配对。除刚度比较外，均限制在弹簧安装。'
        '配对仍可能有不同的未解析缺口，因此是候选选择线索，不是完整路径因果效应估计。'
        '表中n为工况配对数；每种材料只有4个独立地形实现，不能把n当作独立样本数。')
    pair_specs = [('r100−r50', 0, 'r50', 'r100'), ('80°−70°', 1, 'a70', 'a80'),
                  ('5000−2000 N/m', 2, 'spring2000', 'spring5000'),
                  ('p6−p4', 4, 'p4', 'p6'), ('5×2−2×5', 3, '2x5', '5x2')]
    prows = []
    for label, index, first, second in pair_specs:
        values = defaultdict(list)
        for (design, mat, preload, sample), r in keys.items():
            parts = design.split('_')
            if parts[index] != first or (index != 2 and not parts[2].startswith('spring')):
                continue
            parts[index] = second
            q = keys.get(('_'.join(parts), mat, preload, sample))
            if q and r['ok'] and q['ok'] and r['selected_start_xy_m'] == q['selected_start_xy_m']:
                values[mat].append(q['ratio']-r['ratio'])
        prows.append([label, *[f'{median(values[m]):+.4f} (n={len(values[m])})' for m in MATERIALS]])
    put(table(['比较', *[LABELS[m] for m in MATERIALS]], prows))
    put('由此决定：① r100有完成率优势，r50在已完成配对中通常有更高R，两个半径都保留；'
        '② 70–80°是主细化区，新增75°中点，不外推到80°以上；'
        '③ 2000与5000 N/m保留两端，新增3500，并保留1000作为过渡对照；'
        '④ 间距没有单调优势，主表先固定6 mm以衔接现成高完成率锚点，另外保留4/5 mm对照；'
        '⑤ 2×5与5×2未出现跨材料一致优势，必须保留转向对照。'
        '以上是下一轮待检验的设计选择，不是已证明的最优参数。')
    put('100 N/m全库完成率仅6.28%，性能邻域表不再铺开该档全因子扫描，'
        '但论文机制表保留与其他刚度严格配对的4×4构型，不能据完成率认定软弹簧机制无效。'
        '5×5、3×8、8×3的完成率约34–35%，当前数据不足以断言加针有害；'
        '本轮用4×4作规模对照，暂不扩展大阵列。不同针数必须同时报告N和名义占地，不能把固定总P解释成固定每针载荷。')
    put('## 3. 原始轨迹中的具体问题')
    diagnostics = [
        ('r100_a60_fixed0_3x3_p4', 'rough_wall', 1.0, '0'),
        ('r100_a60_fixed0_3x3_p4', 'P40', 1.0, '0'),
        ('r50_a70_fixed0_2x2_p6', 'rough_wall', 1.0, '0'),
        ('r100_a80_spring5000_3x3_p5', 'rough_wall', 1.0, '0'),
    ]
    drows = []
    for key in diagnostics:
        r = keys[key]
        with gzip.open(args.results/r['trace_file'], 'rt', encoding='utf-8') as f:
            trace = json.load(f)
        drag = [s for s in trace['rows'] if s['phase'] == 'drag']
        drows.append((key[0], LABELS[key[1]], f'{r["ratio"]:.3f}',
                      f'{max(s["T_N"] for s in drag):.2f}',
                      f'{max(s["max_rotation_rad"] for s in drag):.3f}',
                      f'{r["resolved_distance_mm"]:.4f}'))
    put('以下均为s000、P=1 N且状态为BALANCED_COMPLETE的实际轨迹：\n\n' +
        table(['设计', '材料', 'R', '采样峰值T（N）', '最大转角（rad）', '已解析（mm）'], drows))
    put('固定分支明确使用小转角线性杆柔度，但上述固定针出现1–3.6 rad量级转角。'
        '这已超出小转角近似适用前提，不能依据这些高值宣称固定安装更优；'
        '也不能由此认定真实针已经断裂。当前折中输出未提供足以认定全程强度合格的证据。'
        '仅减小路径步长不会消除小转角假设本身的适用范围问题。')
    put('刚性针分支也有约402 N的单站尖峰（总预载1 N），所以异常复核不能只盯固定安装。'
        '该尖峰的具体来源尚未定位，不将其直接判成真实承载或直接删掉。'
        '后续检查峰前后接触、分载、Y/Z变化、回缩、重排标记和残差，再做同表面同起点的步长对照。'
        '保留有符号T和原记录，不截峰、不取绝对值、不以收敛残差替代模型适用性。')
    put('## 4. 优先锚点：先看这些设计')
    anchors = ['r100_a70_spring5000_2x2_p6', 'r100_a70_spring2000_2x5_p6',
               'r100_a70_spring5000_3x3_p6', 'r100_a80_spring5000_5x3_p4',
               'r50_a70_spring5000_2x2_p6', 'r50_a70_spring2000_2x5_p6']
    arows = []
    for design in anchors:
        rs = bydesign[design]
        arows.append([design, f'{sum(r["ok"] for r in rs)}/60',
                      *[f'{median([r["ratio"] for r in rs if r["material"] == m and r["ok"]]):.3f}' for m in MATERIALS]])
    put(table(['设计', '完成', *[LABELS[m]+' R' for m in MATERIALS]], arows))
    put('这些是复核锚点，不是最终排名。r100的前四项在60个工况中均到达终点，'
        '但材料内仍有缺口和尖峰；r50配套设计用于防止仅按计算完成率筛掉高阻力小针尖。'
        '优先比较2×2紧凑方案与2×5分载方案；3×3和5×3作为布局补充。')
    established_rows = []
    for design in anchors:
        stats = defaultdict(Counter)
        for r in bydesign[design]:
            if r['status'] == 'EXECUTION_ERROR':
                outcome = None
            else:
                with gzip.open(args.results/r['trace_file'], 'rt', encoding='utf-8') as f:
                    outcome = establishment_from_trace(json.load(f), r['preload_N'])
            stats[r['material']][outcome] += 1
        established_rows.append([design, *[f'{stats[m][True]}/{stats[m][False]}/{stats[m][None]}' for m in MATERIALS]])
    put('### 锚点的实际持续建立\n\n'
        '进一步读取上述6设计的360条工况记录，调用项目现有持续建立协议：T/P≥1、连续0.5 mm、'
        '搜索0–10 mm，不允许掉落。逐区间保留未解析重排缺口，站间采用既定线性插值。'
        '每格为“已观察建立/明确未建立/未知”，分母均为12（3预载×4地形），不代表12个独立地形。'
        '这仅为该协议的离散轨迹后处理，未认证模型有效性；其他11协议未在本轮全库展开。\n\n' +
        table(['设计', *[LABELS[m] for m in MATERIALS]], established_rows))
    put('这项后处理表明，高完成率主要说明队列可计算，尚不能证明已达到既定持续建立目标。'
        '例如两个r100、70°锚点在混凝土上分别为3/0/9和1/1/10；'
        '其未知项必须保留，不能把60/60路径完成写成60/60持续建立。'
        '下一轮优先改善已解析覆盖并比较真实保持段，避免继续只优化平均力。')
    mechanisms=mechanism_designs()
    candidates = fine_designs()
    old = [d for d in candidates if d in bydesign]
    new = [d for d in candidates if d not in bydesign]
    selected_rows = [r for d in old for r in bydesign[d]]
    put('## 5. 保留的对照与邻域参考表（当前不补算、不启动）')
    mechanism_rows=[r for d in mechanisms for r in bydesign[d]]
    mechanism_counts=Counter(r['status'] for r in mechanism_rows)
    put(f'**34设计旧数据机制对照集。** 共{len(mechanism_rows):,}个工况，全部已有粗筛记录。'
        '其状态为：'+'、'.join(f'{s} {n}' for s,n in mechanism_counts.most_common())+'。'
        '按用户最新指示只分析现有路径，不为补齐这张表重跑。'
        '轴向物理范围停止保留为边界；84例硬限位未实现属于历史实现缺口。')
    put(table(['机制对照组','设计数'],Counter(mechanisms.values()).items()))
    put('从保存的宏观逐针`loads_N`可后处理法向分载等效数：'
        '`n_share,n=(Σmax(P_i,0))²/Σmax(P_i,0)²`，全零时未定义；'
        '沿已解析区间按路径长度汇总。现有active字段不能冒充几何接触数；'
        '逐针切向分载、压缩分布和应力的完整路径证据并未在旧折中轨迹中完整保存，'
        '机制代表重算需补足相应输出，不能用末态checkpoint代替全程。')
    put(f'**64设计性能邻域参考子表，当前不执行。** 两张表去重后为{len(set(mechanisms)|set(candidates))}个设计、'
        f'{len(set(mechanisms)|set(candidates))*60:,}个工况；这不是全部需要重新计算的数量。'
        '固定面积加密、固定压力/名义每刺载荷、受控形貌、大阵列和独立测试地形仍是后续组，'
        '当前修复与局部细筛没有覆盖这些论文证据。')
    put(f'共{len(candidates)}个设计：{len(old)}个已有粗筛锚点、{len(new)}个新增中间参数设计。'
        '每设计仍使用P=0.5/1/2 N、五材料、s000–s003，共60个工况；'
        f'全表{len(candidates)*60:,}个工况。新增点仅涉及75°与3500 N/m，'
        '半径、针形、露出规则、弹簧行程、摩擦材料情景、共同背板约束及10 mm路径保持原设定。')
    put(table(['分组', '设计数'], Counter(candidates.values()).items()))
    put('主表以r100为中心，同时保留r50的两角度、两刚度、两布局对照。'
        '该局部表不支持完整估计所有高阶交互，也不能证明50–100 μm之间的全局最优半径。')
    put('### 此前提出的执行顺序与预算（当前不执行）\n\n'
        '1. **针对性复核。** 先处理672条SVD错误所对应的数值原因，确认与候选是否重叠；只重算受影响候选，不重跑全库。'
        '对上节已指出的固定大转角工况保留异常标识，必要时采用项目已有非线性杆实现作有限对照；结果未支持前，固定安装仅作对照。\n'
        '2. **小批步长对照。** 用六个下列条件，各取s000–s003，共24个case，比较25与12.5 μm；'
        '25 μm已有完整且相同配置结果可复用，另算12.5 μm。先判断尖峰、R、正式窗口指标和候选次序是否受步长影响。'
        '不机械加密包络、改变平滑或减小摩擦过渡宽度，避免同时改变多个因素。\n'
        '3. **64设计细筛。** 若上述对照支持保留25 μm，则在相同离散下新增参数点，'
        '并给这52个旧候选中的预算中止case增加计算预算。若对照显示步长影响候选判断，'
        '统一使用12.5 μm另立细筛结果目录，比较组使用同一设置；最多3840个case，不与粗筛覆盖混存。\n'
        '4. **冻结候选。** 按材料分别选出4–8个候选，综合持续建立、J+/J−/Jnet、完成和缺失情况、针数与占地；'
        '本轮只有四个旧地形，是在发现集上的局部细化，不能据此宣布泛化最优。')
    probes = [
        ('r100_a70_spring5000_2x2_p6', 'rough_wall', 1),
        ('r100_a70_spring2000_2x5_p6', 'P40', 2),
        ('r50_a70_spring5000_2x2_p6', 'fired_brick_standard', 1),
        ('r50_a80_spring2000_2x5_p6', 'P100', 2),
        ('r100_a70_spring5000_3x3_p6', 'P240', .5),
        ('r100_a80_spring5000_3x3_p5', 'rough_wall', 1),
    ]
    put(table(['步长对照设计', '材料', 'P（N）', '地形'],
              [(d, LABELS[m], p, 's000–s003') for d, m, p in probes]))
    put('建议细筛CPU预算先采用：弹簧单次180 s、固定单次240 s、含预载换点总计360 s；'
        '这是待执行方案中的计算资源调整，不是放宽残差容差。先用24例的实际耗时校正。'
        '已到达轴向范围的case不靠加预算重复刷过；预载仍按原五候选顺序换点，拖动中途不换点。'
        '预算重算与原结果分别保存，避免旧失败身份被覆盖。'
        '新增75°/3500 N/m与候选子集需实现明确的设计表读取，不能直接将全因子入口指向新档位导致组合意外扩大。')
    sc = Counter(r['status'] for r in selected_rows)
    service = sum(r['solve_s'] for r in selected_rows if np.isfinite(r['solve_s']))
    candidate_errors = sum(r['status'] == 'EXECUTION_ERROR' for r in selected_rows)
    put(f'52个旧候选已有{len(selected_rows):,}条记录：' +
        '、'.join(f'{s} {n}' for s, n in sc.most_common()) + '。'
        f'其中执行错误{candidate_errors}条。若保留25 μm，最少新增参数计算为{len(new)*60}例，'
        f'加上旧候选预算停止{sc["BALANCED_BUDGET_LIMIT"]}例，共{len(new)*60+sc["BALANCED_BUDGET_LIMIT"]}例；'
        '数值修复受影响重算与24例步长对照另计。其余旧结果只在配置和数据有效性适用时复用。')
    mean_service = service/sum(np.isfinite(r['solve_s']) for r in selected_rows)
    projected_h = mean_service*len(candidates)*60/3600
    put(f'旧候选的`solve_s`平均{mean_service:.1f} s，累计{service/3600:.1f} worker小时。'
        f'若全算3840例且保持粗筛耗时分布，约{projected_h:.1f} worker小时，理想16并发约{projected_h/16:.1f}小时。'
        'solve_s是每case求解墙钟时间，非CPU时间；该估计不含准备、尾部等待和提高预算的增量。'
        '若平均每例变成60/120/180 s，则3840例分别为64/128/192 worker小时，'
        '理想16并发为4/8/12小时。实际服务器并发与新步长速度须由小批计时确认，不能把原45 s上限当细筛耗时保证。')
    put('## 6. 细筛必须输出的判断指标')
    put('主筛选保持既定协议：目标T=0.5/1/2 N及T/P≥1，连续保持0.5 mm，'
        '在2/5/10 mm窗口内完成保持，共12协议；沿用原默认不允许掉落的规则。'
        '必须由原始轨迹后处理，当前CSV未提供这些字段；本轮已补算6个锚点的T/P协议，'
        '不能从均值或p90推断其余工况或协议的持续建立。')
    put('完整窗口输出J+/J−/Jnet和持续建立距离S_req；含缺口的窗口正式积分保持未定义，'
        '另列已观察贡献和覆盖率。有完整已解析保持段可报告已观察到建立，'
        '如果此前存在缺口，则最早建立距离仍可能未知；没有观察到建立但搜索窗口不完整时记未知，不能记失败。'
        '事件前后同X点不重复积分；任何持续保持段不得跨未解析重排区间拼接。')
    put('每材料×预载同时报告：已观察建立/明确未建立/未知数量、路径完成率、已解析覆盖、'
        '预算/数值/轴向/执行错误数，以及实际换点比例。对建立率可给“已建立/N”至“(已建立+未知)/N”的识别区间，'
        '该区间不是统计置信区间。排名若随未知结果的处理发生反转，优先补足这些候选，不强行给单一总分。')
    put('对照共享原始地形，逐case核对实际起点。不同预载和设计共用的同一地形不是新增独立重复；'
        '四个地形只展示逐实现结果、方向一致性和范围，不靠数万站点构造很窄的置信区间。'
        '如需总分，沿用项目已声明权重：混凝土1/3、红砖1/3、三种砂纸各1/9；主报告仍按材料分开。')
    put('## 7. 本轮之后的独立确认')
    put('当前服务器沿用“只复用现成地形”，本轮不生成新地形。待局部细筛冻结4–8个候选后，'
        '若另行决定做独立确认，可先增加每材料32个从未参与选型的地形实现，'
        '对应1920–3840个case（候选×5材料×3预载×32实现）。'
        '32是控制成本的第一批建议，不是由现有四个样本估得的充分样本量；届时根据实际区间宽度决定是否增加。'
        '这属于后续新输入方案，需在有CUDA的机器预生成、固定输入后再交服务器；CUDA不可用时不回退CPU。'
        '在此之前不能把旧s000–s003重新编号当验证集，也不直接推进历史256/1024实现或大阵列规划。')
    put('## 8. 2026-09-13细筛前修复与实际复核')
    put('下面是本次会话的代码与有界复核记录，不是运行本分析脚本时重新执行的仿真。'
        '逐一读取5759份轴向停止文件，原因分解为：'
        '**4011例AXIAL_EXTENSION_UNSUPPORTED、1664例EXPOSED_LENGTH_LIMIT、84例COMPRESSION_STOP_NOT_IMPLEMENTED**。'
        '前两类分别是无防拔压缩支路需要轴向拉力、完全缩回边界；不能把它们直接改成成功。'
        '其中84例全部是100 N/m梯度角设计：部分针原露出长度大于4 mm，4 mm处仍有实体硬限位可传力。')
    put('本地求解器已更新为`ijms-balanced-condensed-contact-6`：\n\n'
        '- 按母稿5.8和6.7节补上真实上硬限位：只有smax<l0时激活，'
        '`s=min(-Fa/k,smax)`，额外轴向载荷由限位承担，卸载解除；不截断力、不添加下防拔肩。记录逐针限位反力。\n'
        '- dense SVD抛出LinAlgError时改用LSMR求同一组残差；普通求解缩步后仍停滞时，先试LSMR再进入原重平衡初值搜索。'
        '原1e-3接触/力残差标准不变，记录采用备用求解的站点。\n'
        '- `CondensedArray.run(..., resume=old_result)`允许从已接受checkpoint续接拖动数值/预算中止，'
        '保留原预载、接触、摩擦和全部已接受行；不重新换点，数值失败保留原失败步长。'
        '明确允许重放以前“硬限位未实现”的停止，但禁止绕过完全缩回和轴向拉力范围停止。')
    put(table(['实际工况（均用原25 μm步长）','原停止位置','复核结果','额外求解时间'],[
        ('r50/a60/fixed/3×8/p5，P100，P2，s002','8.035 mm，预算','续算到10 mm','26.70 s'),
        ('r100/a50/k400/2×5/p5，P40，P2，s003','3.365 mm，预算','推进到9.473 mm后数值失败','32.37 s'),
        ('r100/a60/k2000/2×8/p4，红砖，P2，s000','2.487 mm，预算','续算到10 mm','26.89 s'),
        ('r50/a50/k2000/8×2/p4，混凝土，P2，s000','2.863 mm，预算','推进到4.373 mm后数值失败','18.26 s'),
        ('r100/a60/fixed/5×5/p5，P100，P0.5，s002','3.672 mm，数值失败','启用LSMR后续算到10 mm','18.96 s'),
        ('r100/梯度/k100/2×2/p4，P100，P2，s000','1.499 mm，硬限位未实现','从已接受checkpoint续算到10 mm；限位反力最大5.105 N','5.44 s'),
        ('r100/梯度/k100/2×2/p6，P40，P1，s003','2.757 mm，硬限位未实现','推进一小步后仍数值失败','1.04 s')]))
    put('以上续算的原轨迹前缀均逐行保持一致。P100硬限位例从头重跑时另曾在1.436 mm发生数值失败，'
        '因此“从旧接受状态续算通过”不能写成“从头新算已通过”。'
        '服务器SVD错误例r100/a80/k2000/3×3/p6、红砖、P0.5、s003，'
        '在本机原算法下约7.98 s完整跑通，说明该错误未稳定复现。'
        '备用路径通过故障注入测试；不宣称672个历史SVD错误已全部复算消失。')
    put('本次相关测试：`tests/test_balanced_contact.py`与`tests/test_placement_retry.py`共16项通过，'
        '涵盖限位传力与解除、SVD备用算法、连续历史恢复、禁止绕过物理边界以及既有换点规则。'
        '实际小样本已经证明部分停止能修复或补足，但没有据此估计全库恢复比例。'
        '预算放大不能解决全部数值不收敛；当前也仍有大转角、尖峰和未解析重排缺口问题。')
    put('用户最新决定不再补齐粗筛；上述修复和局部复核作为已完成的工程工作保留，不据此启动任何补算批。'
        '后续只在现有数据基础上做候选排除和条件比较。'
        '本机生产仍暂停，服务器包未同步修改，也未启动细筛。新v6不能直接覆盖v5冻结生产目录继续混算。'
        '旧数据只读保留，正式续算器还需按来源模型、配置、原起点和预算分开保存结果。')
    put('## 附：34个机制对照ID\n\n'+table(['序号','分组','设计ID','粗筛完成/60'],
        [(i,group,design,sum(r['ok'] for r in bydesign[design]))
         for i,(design,group) in enumerate(mechanisms.items(),1)]))
    put('## 附：64个细筛候选ID\n\n此表是规划清单，不是已启动队列。新增点以“新增”标注；'
        '旧设计的完成数是全部60个粗筛工况的终点完成数，并不保证10 mm全部解析。')
    put(table(['序号', '分组', '设计ID', '粗筛完成/60'],
              [(i, group, design, str(sum(r['ok'] for r in bydesign[design])) if design in bydesign else '新增')
               for i, (design, group) in enumerate(candidates.items(), 1)]))
    put('复现分析：在仓库根目录运行`.venv/Scripts/python.exe -B scripts/analyze_ijms_coarse.py`。'
        '该分析脚本只读取返回结果并生成报告，不运行求解器。第8节明确区分本次修复复核；'
        '本次没有更改原始结果或粗筛配置，没有启动生产批量。')
    args.report.write_text('\n\n'.join(sections) + '\n', encoding='utf-8')
    write_prototype_selection(rows,args.results,args.report.with_name('IJMS实体样件候选.md'))
    print(json.dumps(dict(report=str(args.report.resolve()), rows=len(rows), candidates=len(candidates),
                          existing=len(old), new=len(new), candidate_status=dict(sc),
                          candidate_mean_s=mean_service, projected_worker_h=projected_h,
                          missing_non_error=sum(r['status'] != 'EXECUTION_ERROR' for r in missing)), ensure_ascii=False))


if __name__ == '__main__':
    main()
