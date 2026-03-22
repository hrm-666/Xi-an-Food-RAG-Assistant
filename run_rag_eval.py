from pathlib import Path

from src.loader import load_tech_docs
from src.splitter import split_documents
from src.database import create_db
from src.qa_chain import get_qa_chain

questions = [
    {
        'title': '1. 信息准确性（基础检索）',
        'question': '西安有哪些典型的本地特色美食？',
        'check': '是否能从文档中准确提取如泡馍、肉夹馍、凉皮等内容，而不是胡编。',
    },
    {
        'title': '2. 细节理解能力（精确定位）',
        'question': '文中提到的“泡馍”有哪些分类？',
        'check': '是否能正确给出“干泡、口汤、水围城、单走”等具体细节，而不是泛泛总结。',
    },
    {
        'title': '3. 总结归纳能力（结构化输出）',
        'question': '请概括文中西安甜品的主要种类。',
        'check': '是否能把柿子饼、甑糕、凉糕等进行归类总结，而不是简单罗列或遗漏。',
    },
    {
        'title': '4. 跨段整合能力（多信息融合）',
        'question': '回民街主要包含哪些区域或街道？',
        'check': '是否能整合多个段落信息（北院门、西羊市、大皮院等），而不是只答一个点。',
    },
    {
        'title': '5. 错误信息识别（鲁棒性 / 幻觉控制）',
        'question': '西安哪家意大利披萨店的元宵个头大、馅料丰富？',
        'check': '是否能识别“意大利披萨店 + 元宵”是不合理组合，并指出问题不成立，再给出文中真实信息（如刘明元宵）。',
    },
]

print('Loading pipeline...')
docs = load_tech_docs('./rag_docs')
chunks = split_documents(docs)
vector_db = create_db(chunks, persist_directory='./vector_db_cleaned')
qa_chain = get_qa_chain(vector_db, chunks)

lines = []
lines.append('# RAG 数据清洗后五个典型问题测试报告')
lines.append('')
lines.append('## 测试说明')
lines.append('')
lines.append('- 测试时间：自动执行')
lines.append('- 测试对象：当前项目中的西安美食 RAG 问答系统')
lines.append('- 文档来源：`rag_docs/西安美食情况的研究.pdf`')
lines.append('- 检索方式：BM25 + MMR 向量检索 + `EnsembleRetriever`')
lines.append('- 向量库状态：使用清洗后文档重建 `vector_db_cleaned/`')
lines.append('')
lines.append('## 测试结果')
lines.append('')

summary = []

for item in questions:
    question = item['question']
    print(f'Running: {question}')
    response = qa_chain.invoke({'query': question})
    answer = response['result'].strip()
    source_docs = response.get('source_documents', [])

    unique_sources = []
    seen = set()
    for doc in source_docs:
        src = doc.metadata.get('source', '未知来源')
        page = doc.metadata.get('page', '?')
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        unique_sources.append(f'{src} 第 {page} 页')

    analysis = '待补充'
    score = '待评估'

    if item['title'].startswith('1.'):
        ok = all(k in answer for k in ['泡馍', '肉夹馍', '凉皮'])
        score = '较好' if ok else '一般'
        analysis = '回答覆盖了题目要求的典型品类，整体基于文档内容作答，未见明显脱离文档的编造。' if ok else '回答能概括部分本地美食，但对典型样例覆盖不够完整。'
    elif item['title'].startswith('2.'):
        keys = ['干泡', '口汤', '水围城', '单走']
        hit = [k for k in keys if k in answer]
        score = '较好' if len(hit) >= 3 else '一般'
        analysis = f'回答命中了 {len(hit)}/4 个关键分类：' + '、'.join(hit) + '。' if hit else '回答没有稳定命中题目要求的关键分类细节。'
    elif item['title'].startswith('3.'):
        keys = ['柿子饼', '甑糕', '凉糕']
        hit = [k for k in keys if k in answer]
        score = '较好' if len(hit) >= 2 else '一般'
        analysis = '回答具备一定归纳能力，能围绕甜品种类进行总结。' if len(hit) >= 2 else '回答偏罗列或覆盖不足，结构化归纳能力一般。'
    elif item['title'].startswith('4.'):
        keys = ['北院门', '西羊市', '大皮院']
        hit = [k for k in keys if k in answer]
        score = '较好' if len(hit) >= 2 else '一般'
        analysis = '回答能够跨段整合多个街区信息，而不是只给一个地点。' if len(hit) >= 2 else '回答的多信息整合能力偏弱，覆盖的街区数量不够。'
    elif item['title'].startswith('5.'):
        reject = ('不' in answer and ('意大利' in answer or '披萨' in answer or '文中未提到' in answer or '问题' in answer))
        real = '刘明' in answer or '元宵' in answer
        score = '较好' if reject and real else ('一般' if reject or real else '较弱')
        analysis = '回答基本识别了错误前提，并尝试回到文中真实信息。' if score == '较好' else '回答对错误前提的识别还不够稳定，幻觉控制能力一般。'

    summary.append((item['title'], score))

    lines.append(f'### {item["title"]}')
    lines.append('')
    lines.append(f'**测试问题：** {question}')
    lines.append('')
    lines.append(f'**检验点：** {item["check"]}')
    lines.append('')
    lines.append('**系统回答：**')
    lines.append('')
    lines.append(answer)
    lines.append('')
    lines.append('**参考来源：**')
    lines.append('')
    if unique_sources:
        for source in unique_sources:
            lines.append(f'- {source}')
    else:
        lines.append('- 未返回来源')
    lines.append('')
    lines.append(f'**效果分析：** {analysis}')
    lines.append('')
    lines.append(f'**单题评价：** {score}')
    lines.append('')

lines.append('## 总体结论')
lines.append('')
lines.append('从这五个典型问题看，当前 RAG 系统在基础事实检索、具体细节定位和多段信息融合上具备一定效果，但最终质量仍明显依赖检索命中情况与生成阶段的表达稳定性。')
lines.append('')
for title, score in summary:
    lines.append(f'- {title}：{score}')
lines.append('')
lines.append('更适合的使用场景是围绕文档内容进行事实型问答与摘要，不太适合脱离文档前提的开放式推断。')
lines.append('')

Path('RAG数据清洗后五个典型问题测试报告.md').write_text('\n'.join(lines), encoding='utf-8')
print('Report written to RAG数据清洗后五个典型问题测试报告.md')
