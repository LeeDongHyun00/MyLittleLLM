"""Layout-aware policy passages and budgeted retrieval with lexical retention.

PDF parsing is offline. The online path keeps exact source text and verified
layout scope; it never uses reference answers or arbitrary confidence cutoffs.
"""
from collections import Counter
import json
from pathlib import Path
import re
import time

from .hierarchical import bm25_ranks, scope_matches, sha
from .retrieval import ROOT, ambiguous_comparison, read_jsonl, retrieve


def load_config():
    return json.loads((ROOT/'config/preprocessing.json').read_text())


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(ROOT/load_config()['tokenizer']))


def token_count(text, tok):
    return len(tok.encode(text, add_special_tokens=False).ids)


def track_for_page(source_id, page, config):
    for entry in config['track_ranges'].get(source_id, []):
        if entry['pages'][0] <= page <= entry['pages'][1]:
            return entry['track']
    return None


def question_track(question, explicit=None):
    allowed={'논술','학생부교과','학생부종합','실기/실적'}
    if explicit is not None:
        if explicit not in allowed:raise ValueError('Unsupported admission track')
        return explicit
    matches=[t for t in allowed if t in question]
    return matches[0] if len(matches)==1 else None


def contextual_text(row):
    scope=f"{row['institution']} | {row['year']}학년도 | PDF {row['pdf_page']}쪽"
    if row.get('admission_track'):scope+=' | 전형 계열: '+row['admission_track']
    if row.get('section_semester'):scope+=f" | 적용 학기: {row['section_semester']}학기"
    return scope+'\n'+row['source_text']


def focused_query(question, year, institution):
    """Remove already-filtered scope boilerplate, retaining conditions/numbers."""
    text=re.sub(rf'{year}\s*학년도|{year}\s*년', ' ', question)
    aliases={'부산대학교':['부산대학교','부산대'],'서울대학교':['서울대학교','서울대'],
             '경희대학교':['경희대학교','경희대'],'한국장학재단':['한국장학재단']}
    for name in aliases.get(institution,[institution] if institution else []):text=text.replace(name,' ')
    for name in ['주거안정장학금','수시 모집요강','수시모집요강','시행계획']:text=text.replace(name,' ')
    return text.strip() or question


def clause_blocks(text, tok, target):
    """Pack complete numbered clauses; attach notes to the preceding clause.

    Oversize atomic clauses remain intact (and may be skipped at retrieval).
    This deliberately avoids token-slicing away a trailing exception.
    """
    lines=[s.strip() for s in text.splitlines() if s.strip()]
    atoms=[]
    current=[]
    for line in lines:
        boundary=bool(re.match(r'^(?:[□◦○]|[가-하][.)]\s|\d{1,2}[.)]\s|\d{1,2}\s{2,})',line))
        if boundary and current:
            atoms.append('\n'.join(current));current=[]
        current.append(line)
    if current:atoms.append('\n'.join(current))
    packed=[];current=''
    for atom in atoms:
        proposed=current+'\n'+atom if current else atom
        if current and token_count(proposed,tok)>target:
            packed.append(current);current=atom
        else:current=proposed
    if current:packed.append(current)
    return packed


def expanded_table_rows(table, page):
    """Repeat merged cells using their bounding boxes, never blind forward-fill.

    A None slot has no inherited value unless a real spanning cell covers it.
    The original bbox of every expanded cell is retained for auditing.
    """
    cells=table.cells
    xs=sorted(set(v for c in cells for v in (c[0],c[2])))
    ys=sorted(set(v for c in cells for v in (c[1],c[3])))
    cache={}
    result=[]
    for y0,y1 in zip(ys,ys[1:]):
        row=[]
        for x0,x1 in zip(xs,xs[1:]):
            mx,my=(x0+x1)/2,(y0+y1)/2
            match=next((c for c in cells if c[0]<=mx<c[2] and c[1]<=my<c[3]),None)
            if match is None:
                row.append({'text':'','bbox':None});continue
            if match not in cache:cache[match]=page.crop(match).extract_text() or ''
            row.append({'text':cache[match],'bbox':list(match)})
        result.append(row)
    return result


def build_index(output=None):
    import pdfplumber
    output=Path(output or ROOT/'outputs/preprocessed')
    if output.exists():raise ValueError('Choose a new output directory; do not overwrite an index')
    config=load_config();tok=tokenizer()
    sources={r['source_id']:r for r in read_jsonl(ROOT/'data/sources.jsonl')}
    snapshots={r['source_id']:r for r in read_jsonl(ROOT/'data/retrieval_snapshot.jsonl')}
    corpus=json.loads((ROOT/'config/corpus.json').read_text())['sources']
    rows=[];audit=[]
    start=time.monotonic()
    for source_id,policy in corpus.items():
        snap=snapshots[source_id];source=sources[source_id]
        if sha(ROOT/snap['raw_path'])!=snap['sha256']:raise ValueError('Source hash changed')
        annotated=config.get('annotation_source_sha256',{}).get(source_id)
        if annotated and annotated!=snap['sha256']:raise ValueError('Track annotations require source re-review')
        pages=read_jsonl(ROOT/snap['extracted_path'])
        pdf=pdfplumber.open(ROOT/snap['raw_path']) if source_id in config['pdfplumber_sources'] else None
        for old in pages:
            n=old['pdf_page']
            if not any(a<=n<=b for a,b in policy.get('page_ranges',[[1,len(pages)]])):continue
            if ambiguous_comparison(old['text']):
                audit.append({'source_id':source_id,'pdf_page':n,'status':'multi_year_quarantine'});continue
            meta={'source_id':source_id,'pdf_page':n,'year':source['academic_year'],
                  'semester':source['semester'],'task':policy['task'],'institution':policy['institution'],
                  'sha256':snap['sha256'],'url':source['landing_url'],
                  'admission_track':track_for_page(source_id,n,config),
                  'section_semester':None,'scope_review_status':config['review_status']}
            page=pdf.pages[n-1] if pdf else None
            text=(page.extract_text(layout=False) or '') if page else old['text']
            if len(text.strip())<40:
                audit.append({'source_id':source_id,'pdf_page':n,'status':'low_text_requires_ocr'});continue
            special=config['reviewed_schedule']
            is_schedule=source_id==special['source_id'] and n==special['pdf_page']
            if not is_schedule:
                for i,block in enumerate(clause_blocks(text,tok,config['paragraph_target_tokens'])):
                    row=dict(meta,chunk_id=f'{source_id}:p{n}:block{i}',kind='clause',
                             source_text=block,extraction='pdfplumber' if page else 'original_text',bbox=None)
                    row['text']=contextual_text(row);rows.append(row)
            tables=page.find_tables() if page else []
            accepted=0
            for ti,table in enumerate(tables):
                raw=table.extract()
                if len(raw)<2 or len(raw[0])<2:continue
                if is_schedule and (snap['sha256']!=special['sha256'] or len(raw)!=special['rows'] or
                        any(abs(a-b)>.1 for a,b in zip(table.bbox,special['table_bbox']))):
                    raise ValueError('Reviewed table geometry changed; re-review before using scope annotations')
                expanded=expanded_table_rows(table,page)
                if is_schedule and len(expanded)!=special['rows']:
                    raise ValueError('Merged row reconstruction changed')
                header=expanded[0]
                # Do not pretend that the first data row of a headerless table
                # is a header and repeat its values as labels on later rows.
                header_terms=r'대상|구분|전형|제출|내용|일정|모집|대학|성적|기준|항목|수량|비고|반영|과목|단계|학년|지원횟수'
                header_hits=sum(bool(re.search(header_terms,re.sub(r'\s+','',c['text'])))
                                and len(c['text'])<100 for c in header)
                if not is_schedule and header_hits<2:continue
                page_notes='\n'.join(line.strip() for line in text.splitlines() if line.strip().startswith('※'))
                for ri,cells in enumerate(expanded[1:],1):
                    body=[];seen=set()
                    for ci,cell in enumerate(cells):
                        if not cell['text'].strip() or tuple(cell['bbox'] or []) in seen:continue
                        seen.add(tuple(cell['bbox'] or []))
                        label=header[ci]['text'].replace('\n',' ') if ci<len(header) else ''
                        body.append(f"{label or ('열 '+str(ci+1))}: {cell['text']}")
                    if not body:continue
                    section_semester=None
                    if is_schedule:
                        for entry in special['semester_row_ranges']:
                            if entry['rows'][0]<=ri<=entry['rows'][1]:section_semester=entry['semester']
                    row=dict(meta,chunk_id=f'{source_id}:p{n}:table{ti}:row{ri}',kind='table_row',
                             section_semester=section_semester,source_text='\n'.join(body)+
                             ('\n페이지 주의사항(행별 적용 확인 필요):\n'+page_notes if page_notes else ''),
                             bbox=list(table.bbox),cells=cells,extraction='pdfplumber_merged_cell_geometry')
                    row['text']=contextual_text(row);rows.append(row)
                accepted+=1
            if is_schedule and accepted!=1:raise ValueError('Reviewed schedule table not recovered')
            audit.append({'source_id':source_id,'pdf_page':n,'status':'parsed','tables':accepted})
            if page:page.close()
        if pdf:pdf.close()
        print(json.dumps({'source':source_id,'total_units':len(rows)},ensure_ascii=False),flush=True)
    output.mkdir(parents=True)
    (output/'index.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    manifest={'config':config,'units':len(rows),'elapsed_seconds':round(time.monotonic()-start,3),
              'index_sha256':sha(output/'index.jsonl'),'config_sha256':sha(ROOT/'config/preprocessing.json'),
              'builder_sha256':sha(Path(__file__)),'tokenizer_sha256':sha(ROOT/config['tokenizer']),
              'source_hashes':{s:snapshots[s]['sha256'] for s in corpus},'page_audit':audit,
              'pdfplumber_version':pdfplumber.__version__,
              'limits':['No OCR performed; low text pages require review.',
                        'Table structure extracted automatically except reviewed housing schedule scope.',
                        'Admission metadata covers reviewed 2026 PNU track families, not every subtrack.',
                        'Page-wide notes are retained but not automatically resolved to individual rows.']}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


def scope_compatible(row, *, year, task, institution, semester, track):
    effective=dict(row,semester=row.get('section_semester') or row['semester'])
    if not scope_matches(effective,year,task,institution,semester):return False
    if semester and row.get('section_semester') not in (None,semester):return False
    if track and row.get('admission_track') not in (None,track):return False
    return True


class PreprocessedRetriever:
    def __init__(self, semantic=None, path=None):
        path=Path(path or ROOT/'outputs/preprocessed')
        self.manifest=json.loads((path/'manifest.json').read_text())
        self.config=load_config()
        if self.manifest['config_sha256']!=sha(ROOT/'config/preprocessing.json'):
            raise ValueError('Preprocessing config changed; rebuild index')
        if self.manifest['index_sha256']!=sha(path/'index.jsonl'):raise ValueError('Index changed')
        self.rows=read_jsonl(path/'index.jsonl')
        self.legacy=read_jsonl(ROOT/'outputs/index.jsonl')
        self.semantic=semantic
        self.tok=tokenizer()

    def retrieve(self, question, *, year, task, institution=None, semester=None, admission_track=None,
                 semantic_result=None, context_tokens=None):
        start=time.monotonic()
        if task=='admissions' and not institution:raise ValueError('Institution is required')
        if task=='housing' and semester is None:raise ValueError('Semester is required')
        track=question_track(question,admission_track)
        scope=dict(year=year,task=task,institution=institution,semester=semester,track=track)
        units=[r for r in self.rows if scope_compatible(r,**scope)]
        budget=context_tokens or self.config['context_tokens']
        focus=focused_query(question,year,institution)
        ranked=[units[i] for i in bm25_ranks(units,focus)[:24]] if units else []
        legacy=[]
        special=self.config['reviewed_schedule']
        raw_scoped=[]
        for r in self.legacy:
            meta=dict(r,admission_track=track_for_page(r['source_id'],r['pdf_page'],self.config))
            # Replace this mixed-scope page with reviewed rows, never feed its flat text.
            if r['source_id']==special['source_id'] and r['pdf_page']==special['pdf_page']:continue
            if scope_compatible(meta,**scope):raw_scoped.append(meta)
        for r in retrieve(raw_scoped,question,year=year,task=task,institution=institution,semester=semester,k=8):
            row=dict(r,source_text=r['text'],section_semester=None,kind='legacy_retained')
            row['text']=contextual_text(row);legacy.append(row)
        semlog={};sem=[]
        if semantic_result is not None:
            _,semlog=semantic_result
        elif self.semantic is not None:
            _,semlog=self.semantic.retrieve(question,year=year,task=task,institution=institution,semester=semester)
        candidate_pages=[]
        for candidate in semlog.get('candidates',[]):
            match=re.match(r'^(.*):p(\d+):',candidate['chunk_id'])
            if match:
                key=(match[1],int(match[2]))
                if key not in candidate_pages:candidate_pages.append(key)
        # All eight semantic candidates survive into the pool. Rank complete
        # structural units on these pages with the same lexical scorer.
        semunits=[r for r in units if (r['source_id'],r['pdf_page']) in candidate_pages]
        sem=[semunits[i] for i in bm25_ranks(semunits,focus)[:16]] if semunits else []
        # Reserve lexical opportunities instead of allowing a neural reranker to
        # delete every exact-term match. No gold-driven choice or truncation.
        pools=[legacy,ranked,sem]
        ordered=[]
        for i in range(max(map(len,pools),default=0)):
            for name,pool in zip(['legacy','structured','semantic_pages'],pools):
                if i<len(pool):ordered.append((name,pool[i]))
        evidence=[];used=set();characters=[];skipped=[]
        for method,row in ordered:
            if row['chunk_id'] in used:continue
            normalized=re.sub(r'\s+','',row['source_text'])
            # A longer clause may add an exception to an already selected
            # fragment. Only discard a candidate wholly contained in old text.
            if any(normalized in old for old in characters):continue
            size=token_count(row['text'],self.tok)
            if size>budget:
                skipped.append({'chunk_id':row['chunk_id'],'reason':'whole_unit_exceeds_remaining_budget','tokens':size});continue
            evidence.append(dict(row,retrieval_method=method))
            characters.append(normalized);used.add(row['chunk_id']);budget-=size
            if len(evidence)>=self.config['top_k']:break
        return evidence,{'seconds':round(time.monotonic()-start,4),'context_tokens':(context_tokens or self.config['context_tokens'])-budget,
                         'admission_track':track,'semantic_enabled':bool(self.semantic or semantic_result is not None),
                         'focused_query':focus,
                         'candidate_pages':candidate_pages,'skipped':skipped,
                         'method':'lexical_retention_structural_units_semantic_candidate_pages',
                         'scope_limit':'reviewed track families and housing schedule rows; other scopes remain document-level'}
