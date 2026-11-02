"""Freeze a source-checked, never-trained rule holdout before model evaluation.

This is an assistant-authored research test, not independently human-approved
gold. Inputs, answers, protocol, model hashes and retrieval are frozen together.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mylittlellm.distillation_data import normalize, sha256_file
from mylittlellm.retrieval import ROOT, read_jsonl, retrieve


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def seed_cases():
    cases = []
    def add(id, group, year, task, question, choices, evidence, gold, **extra):
        cases.append(dict(id=id, leakage_group=group, year=year, task=task, question=question,
                          correct_text=choices[0], choices=choices, evidence_keys=evidence,
                          gold_answer=gold, **extra))
    add('A01','pnu-application-deadline',2025,'admissions',
        '2025학년도 부산대 수시 인터넷 원서접수의 최종 마감 날짜와 시각은 언제인가요?',
        ['2024년 9월 13일 18시','2024년 9월 19일 18시','2025년 9월 12일 17시','2024년 9월 13일 17시'],
        ['deadline25','deadline26'],'2024년 9월 13일 18시입니다.')
    add('A02','pnu-application-deadline',2026,'admissions',
        '2026학년도 부산대 수시 인터넷 원서접수의 최종 마감 날짜와 시각은 언제인가요?',
        ['2025년 9월 12일 17시','2025년 9월 15일 17시','2024년 9월 13일 18시','2025년 9월 12일 18시'],
        ['deadline26','deadline25'],'2025년 9월 12일 17시(오후 5시)입니다.', free=True,
        required=[r'2025',r'9\s*(?:월|[./-])\s*12',r'17\s*(?::\s*00|시)|오후\s*5\s*시'], forbidden=[r'18\s*시|오후\s*6\s*시'])
    add('A03','pnu-postpayment-changes',2025,'admissions',
        '2025학년도 부산대 수시에서 전형료를 결제한 뒤 모집단위 변경과 연락처 오타 수정은 각각 가능한가요?',
        ['모집단위 변경은 불가하고 연락처는 공지에 따라 수정 신청 가능','모집단위와 연락처 모두 자유롭게 변경 가능','모집단위와 연락처 모두 수정 신청 불가','모집단위만 변경 가능하고 연락처는 수정 불가'],
        ['edit25','edit26'],'전형·모집단위 변경은 불가하나 연락처 등 인적사항은 공지를 참고하여 수정 신청할 수 있습니다.')
    add('A04','pnu-postpayment-changes',2026,'admissions',
        '2026학년도 부산대 수시 원서 결제 후 주소 오타를 발견했습니다. 주소와 지원 모집단위를 고칠 수 있는지 구분해 주세요.',
        ['주소는 공지에 따라 수정 신청 가능하고 모집단위 변경은 불가','주소와 모집단위 모두 변경 불가','주소와 모집단위 모두 자유롭게 변경 가능','모집단위만 변경 가능하고 주소는 수정 불가'],
        ['edit26','edit25'],'주소 등 인적사항은 입학홈페이지 공지를 참고하여 수정 신청할 수 있으나 모집단위는 변경할 수 없습니다.', free=True,
        required=[r'주소|인적사항',r'수정\s*신청|신청.{0,12}수정',r'모집단위.{0,30}(불가|없|못)'], forbidden=[r'모집단위.{0,12}(자유롭게|변경\s*가능)'])
    add('A05','pnu-document-delivery',2025,'admissions',
        '2025학년도 부산대 수시 제출서류를 2024년 9월 19일 소인이 찍힌 등기우편으로 보냈고 다음 날 도착합니다. 다른 요건을 충족하면 소인 기준의 제출기한 요건을 충족하나요?',
        ['9월 19일 소인까지 유효하므로 충족','9월 19일 도착분만 유효하므로 미충족','9월 13일 소인까지여서 미충족','방문 제출만 인정하므로 미충족'],
        ['mail25','mail26'],'2024년 9월 19일 소인까지 유효하므로 제시한 소인 기준을 충족합니다.')
    add('A06','pnu-document-delivery',2026,'admissions',
        '2026학년도 부산대 수시 서류를 마감일에 직접 방문 제출하려 합니다. 허용되는 제출 방식과 우편 소인 마감일은 무엇인가요?',
        ['방문 불가, 등기우편이며 2025년 9월 15일 소인까지','방문 가능, 우편은 2025년 9월 12일 소인까지','방문과 일반우편 가능, 2025년 9월 19일 소인까지','등기우편만 가능, 2024년 9월 19일 소인까지'],
        ['mail26','mail25'],'방문 제출은 불가하고 등기우편만 가능하며 2025년 9월 15일 소인까지 유효합니다.')
    add('A07','pnu-document-issue-date',2026,'admissions',
        '2026학년도 부산대 수시 일반 제출서류의 원본을 2025년 9월 2일 발급받았습니다. 저소득층 증빙은 아닙니다. 발급일 기준만 충족하나요?',
        ['일반 서류는 2025년 9월 1일부터이므로 충족','모든 서류가 9월 5일부터여서 미충족','모든 서류가 원서접수 당일 발급이어야 하므로 미충족','제출서류는 원본이어도 인정하지 않으므로 미충족'],
        ['issue26'],'일반 제출서류는 2025년 9월 1일부터 발급한 원본을 인정하므로 해당 기준을 충족합니다.')
    add('A08','pnu-document-issue-date',2026,'admissions',
        '2026학년도 부산대 수시 저소득층학생 증빙 원본을 2025년 9월 2일 발급받았습니다. 발급일 기준을 충족하나요? 인정되는 시작일도 알려주세요.',
        ['충족하지 못하며 저소득층 서류는 2025년 9월 5일부터','충족하며 모든 서류는 2025년 9월 1일부터','충족하며 저소득층 서류에는 발급일 제한이 없음','충족하지 못하며 저소득층 서류는 2025년 9월 15일부터'],
        ['issue26'],'충족하지 못합니다. 저소득층학생 제출서류는 2025년 9월 5일부터 발급한 원본만 인정합니다.', free=True,
        required=[r'불가|못|않|미충족|아니',r'2025',r'9\s*(?:월|[./-])\s*5'], forbidden=[r'9\s*월\s*1\s*일.{0,12}(충족|가능)'])
    add('A09','national-susi-application-count',2025,'admissions',
        '2025학년도 수시에서 횟수 제한의 예외에 해당하지 않는 동일 대학의 서로 다른 전형 두 개에 지원했습니다. 6회 제한에서 몇 회로 세나요?',
        ['전형별로 세므로 2회','대학이 같으므로 1회','같은 대학의 전형은 횟수에서 제외','등록한 뒤에만 횟수로 계산'],
        ['six25','six26'],'동일 대학에 복수 지원해도 지원한 전형마다 각각 산정하므로 2회입니다.')
    add('A10','national-susi-application-count',2026,'admissions',
        '2026학년도 일반적인 수시 6회 제한 대상자가 일곱 번째 원서를 접수했습니다. 원서접수 시간 기준 일곱 번째 접수는 어떻게 처리되나요?',
        ['6회 이후 접수이므로 취소','같은 대학이면 언제나 유지','추가 전형료만 내면 유지','합격하면 일곱 번째 접수도 유효'],
        ['six26','six25'],'원서 접수 시간을 기준으로 6회 이후의 접수는 취소합니다.')
    add('A11','pnu-comprehensive-cross-application',2025,'admissions',
        '각 전형 지원자격과 전국 수시 횟수 제한을 충족한다고 가정합니다. 2025학년도 부산대 학생부종합 사회배려자전형과 저소득층학생전형 두 곳에 복수지원할 수 있나요?',
        ['해당 학생부종합 유형에서 2회 복수지원 가능','두 전형 간 복수지원은 무조건 금지','학생부종합은 전형 수와 관계없이 6회 지원 가능','두 전형 모두 일반 수시 횟수에서 제외'],
        ['multi25','multi26'],'2025학년도 해당 학생부종합 유형에서는 2회 복수지원할 수 있습니다. 각 지원자격과 전국 횟수 제한은 별도 충족해야 합니다.')
    add('A12','pnu-comprehensive-cross-application',2026,'admissions',
        '각 전형 지원자격을 충족한다고 가정합니다. 2026학년도 부산대 학생부종합 사회배려자전형과 저소득층학생전형에 동시에 복수지원할 수 있나요?',
        ['두 전형이 속한 묶음 안에서는 복수지원 불가','2025학년도처럼 두 전형에 동시에 지원 가능','지원자격만 충족하면 학생부종합 전형 모두 지원 가능','같은 모집단위를 고르면 복수지원 가능'],
        ['multi26','multi25'],'불가능합니다. 2026학년도에는 사회배려자·저소득층학생 등이 속한 학생부종합 전형 묶음 안에서 복수지원할 수 없습니다.', free=True,
        required=[r'불가|불가능|못|않',r'사회배려|저소득층'], forbidden=[r'동시.{0,8}지원\s*가능'])
    add('A13','pnu-exam-procedure',2026,'admissions',
        '2026학년도 부산대 대학별 고사에 수험표 인쇄물을 가져오지 않았습니다. 신분증 등 다른 사항은 갖췄습니다. 수험표만 놓고 어떻게 해야 하나요?',
        ['해당 고사장 본부에서 출력하여 응시 가능','수험표 미지참자는 예외 없이 불합격','휴대전화 사진만 있으면 모든 절차가 면제','응시 대신 다음 해 시험으로 자동 이월'],
        ['exam26'],'고사 당일 수험표 미지참자는 해당 고사장 본부에서 수험표를 출력하여 응시할 수 있습니다.')
    add('A14','pnu-exam-procedure',2026,'admissions',
        '2026학년도 부산대 대학별 고사에서 스마트워치 전원을 끄고 소지만 하면 괜찮나요?',
        ['소지 자체가 금지되므로 안 됨','전원을 끄면 소지 가능','통신 기능만 끄면 소지 가능','시험 중 사용하지 않았다고 말하면 항상 허용'],
        ['exam26'],'스마트워치 등 전자기기는 소지 자체가 금지됩니다.')
    add('A15','pnu-rural-residence',2026,'admissions',
        '2026학년도 부산대 농어촌학생 일반적인 6년과정 유형의 거주 요건은 누구에게 적용되나요? 별도의 가족관계 예외는 묻지 않습니다.',
        ['중학교 입학부터 고교 졸업까지 본인과 부모 모두','초등학교 입학부터 고교 졸업까지 부모만','고등학교 재학 3년 동안 본인만','원서 접수일 하루 동안 본인과 부모 모두'],
        ['rural26'],'6년과정은 중학교 입학일부터 고교 졸업일까지 재학기간 중 본인과 부모 모두의 농어촌 거주를 요구합니다.')
    add('A16','pnu-rural-residence',2026,'admissions',
        '2026학년도 부산대 농어촌학생 12년과정에서 초·중·고 전 교육과정을 농어촌 학교에서 이수하고 그 기간 내내 본인이 농어촌에 살았습니다. 이 유형의 명시된 거주 요건은 부모에게도 요구되나요?',
        ['12년과정은 본인의 거주를 요구하고 부모 거주는 명시하지 않음','6년과정과 마찬가지로 부모 모두의 거주를 요구','부모 한 명만 거주하면 본인 거주는 불필요','원서접수 직전에만 본인이 거주하면 충분'],
        ['rural26'],'12년과정의 명시된 거주 요건은 초등학교 입학부터 고교 졸업까지 본인에게 적용됩니다. 부모의 거주 요건은 이 유형에 명시하지 않습니다.', free=True,
        required=[r'본인',r'부모.{0,40}(않|없|불필요|요구하지|명시하지)'], forbidden=[r'부모\s*모두.{0,12}(필요|요구합니다)'])
    add('S01','national-I-income-range',2025,'scholarship',
        '2025년 2월 국가장학금 기본계획의 Ⅰ유형 지원구간 기준만 묻습니다. 9구간 대학생은 확대된 소득구간 범위에 들어가나요?',
        ['9구간 이하로 확대되어 범위에 포함','8구간 이하만 인정되어 제외','10구간만 인정되어 제외','소득구간 기준 자체가 폐지됨'],
        ['income25'],'Ⅰ유형은 9구간 이하로 확대되어 9구간도 소득구간 범위에 포함됩니다. 최종 수혜의 다른 요건은 별도입니다.')
    add('S02','national-I-income-range',2025,'scholarship',
        '2025년 2월 기본계획의 국가장학금 Ⅰ유형 지원구간 기준만 묻습니다. 10구간은 Ⅰ유형의 명시된 범위에 포함되나요?',
        ['9구간 이하이므로 Ⅰ유형의 범위 밖','9구간과 10구간은 모두 Ⅰ유형 범위 안','Ⅱ유형의 예외가 있으므로 Ⅰ유형도 자동 포함','성적이 좋으면 지원구간 요건이 자동 삭제'],
        ['income25'],'Ⅰ유형의 명시된 소득구간 범위는 9구간 이하이므로 10구간은 범위 밖입니다.')
    add('S03','national-II-participation-and-priority',2025,'scholarship',
        '2025년 2월 계획에서 국가장학금 Ⅱ유형 대학 참여조건 완화의 교내장학금 규모 기준은 전년 대비 몇 % 이상인가요?',
        ['90% 이상이며 등록금 동결·인하 및 총액 대비 감면율 조건도 있음','80% 이상이면 등록금을 올려도 무조건 참여','100% 이상만 허용하고 완화 없음','50% 이상이며 다른 조건은 없음'],
        ['typeII25'],'교내장학금 규모는 전년 대비 90% 이상입니다. 등록금 동결·인하와 등록금 총액 10% 이상의 감면율 조건도 확인해야 합니다.')
    add('S04','national-II-participation-and-priority',2025,'scholarship',
        '2025년 2월 기본계획에서 보건복지부 가족돌봄청년 전담 지원 사업 대상 대학생이 국가장학금 Ⅱ유형 우선지원 대상에 포함되나요? 이것만으로 지급이 확정되나요?',
        ['우선지원 대상에 포함되지만 이 문구만으로 지급 확정은 아님','우선지원 대상에서 제외됨','Ⅰ유형에만 포함되고 Ⅱ유형은 제외','신청과 심사 없이 전액 지급이 자동 확정'],
        ['typeII25'],'가족돌봄청년은 Ⅱ유형 우선지원 대상에 포함됩니다. 우선지원 문구만으로 개인의 지급이 확정됐다고 판단할 수는 없습니다.', free=True,
        required=[r'우선\s*지원|우선지원',r'포함|대상',r'확정.{0,25}(않|없|아니)|보장.{0,20}(않|없)|심사|자동.{0,15}(않|아니)'], forbidden=[r'자동.{0,10}(확정됩니다|지급됩니다)'])
    add('S05','national-work-hourly-pay',2025,'scholarship',
        '2025년 2월 근로장학금 기본계획에 나온 인상 후 교내근로 시간당 지원단가는 얼마인가요?',
        ['10,030원','9,860원','12,430원','12,220원'],['work25'],'교내근로 시간당 지원단가는 10,030원입니다.')
    add('S06','national-work-hourly-pay',2025,'scholarship',
        '2025년 2월 근로장학금 기본계획에 나온 인상 후 교외근로 시간당 지원단가는 얼마인가요?',
        ['12,430원','12,220원','10,030원','9,860원'],['work25'],'교외근로 시간당 지원단가는 12,430원입니다.', free=True,
        required=[r'12,?430|1만\s*2,?430|1만\s*2430'], forbidden=[r'12,?220|10,?030|9,?860'])
    add('S07','dream-stage-specific-support',2025,'scholarship',
        '2025년 2월 드림장학금 계획에서 해외유학을 준비하는 저소득층 고2~3 유학준비생의 지원 항목은 무엇인가요?',
        ['학업장려비','국내 대학의 등록금만','유학준비생에게는 아무 지원도 없음','이미 해외대학 졸업한 사람의 취업수당'],['dream25'],'고2~3 유학준비생에게는 학업장려비를 지원합니다.')
    add('S08','dream-stage-specific-support',2025,'scholarship',
        '2025년 2월 드림장학금 계획에서 장학생이 해외대학에 진학한 유학생 단계의 지원 항목은 무엇인가요?',
        ['학비와 체재비 등','고교 교복비만','학업장려비만 있고 학비는 지원하지 않음','국내 대학 기숙사비만'],['dream25'],'해외대학에 진학한 유학생에게는 학비와 체재비 등을 지원합니다.')
    add('U01','unsupported-target-year',2027,'admissions',
        '제공된 2025·2026학년도 부산대 자료만으로 2027학년도 수시 원서접수 최종 마감일과 시각을 확정해 주세요.',
        ['2027학년도 자료가 없어 확정할 수 없음','2025년 9월 12일 17시로 확정','2026년 9월 12일 17시로 확정','2026년 9월 13일 18시로 확정'],['deadline25','deadline26'],'2027학년도 근거가 없으므로 확정할 수 없습니다.',expected_action='abstain')
    add('U02','unsupported-target-year',2026,'scholarship',
        '제공된 2025년 2월 근로장학금 계획만으로 2026년 교외근로 시간당 단가를 확정할 수 있나요?',
        ['2026년 근거가 없어 확정할 수 없음','12,430원으로 확정','10,030원으로 확정','2025년 단가에 무조건 10% 인상한 금액으로 확정'],['work25'],'2026년 단가를 확정할 근거가 없습니다.',expected_action='abstain')
    add('U03','individual-outcome-uncertainty',2026,'admissions',
        '내신 평균이 2.1등급입니다. 제공된 부산대 수시 원서접수 일정만으로 제 합격 확률을 정확한 숫자로 알려주세요.',
        ['개인 합격 확률을 산출할 근거가 부족함','합격 확률 100%','합격 확률 50%','합격 확률 0%'],['deadline26'],'원서접수 일정만으로 개인의 합격 확률을 산출할 수 없습니다.',expected_action='abstain')
    add('U04','missing-academic-year',None,'admissions',
        '부산대 수시 원서접수 마감일과 시각을 알려주세요. 지원할 학년도는 아직 말씀드리지 않았습니다.',
        ['지원 학년도를 먼저 확인해야 함','항상 9월 13일 18시라고 답함','항상 9월 12일 17시라고 답함','원서접수에는 마감 시각이 없다고 답함'],['deadline25','deadline26'],'지원할 학년도를 먼저 확인해야 합니다.',expected_action='clarify',free=True,
        required=[r'학년도',r'확인|알려|어느|몇|필요|명시'], forbidden=[r'마감.{0,10}(입니다|확정)'])
    return cases


# Exact normalized-text selectors; no answer text is inserted into the model context.
SELECTORS = {
 'deadline25':('pusan-2025-susi',21,'1. 원서 접수 일정','2. 원서 접수 방법'),
 'deadline26':('pusan-2026-susi',21,'1 원서 접수 기간','2 원서 접수 방법'),
 'edit25':('pusan-2025-susi',21,'마. 원서 접수 완료','사. 원서 접수 후'),
 'edit26':('pusan-2026-susi',21,'다) 원서 접수 완료','마) 원서 접수 후'),
 'mail25':('pusan-2025-susi',22,'2. 서류는 등기우편','3. 모든 제출서류'),
 'mail26':('pusan-2026-susi',23,'2 서류는 등기우편','3 서류도착 확인'),
 'issue26':('pusan-2026-susi',23,'1 제출서류는','2 서류는 등기우편'),
 'six25':('pusan-2025-susi',25,'1. 수시모집 지원 횟수','2. 복수지원 금지'),
 'six26':('pusan-2026-susi',27,'1 수시모집 지원 횟수','2 복수지원 금지'),
 'multi25':('pusan-2025-susi',26,'가. 학생부교과(학생부교과전형)','다. 논술(논술전형)'),
 'multi26':('pusan-2026-susi',28,'(1) 학생부교과 유형','(4) 논술(논술전형)'),
 'exam26':('pusan-2026-susi',29,'나) 고사 당일 수험표','라) 기상이변'),
 'rural26':('pusan-2026-susi',111,'6년과정 이수자','재학사항 구분'),
 'income25':('moe-2025-scholarship-plan',3,'□ 국가장학금 Ⅰ유형 지원 대상 확대','〈 참고'),
 'typeII25':('moe-2025-scholarship-plan',4,'□ 국가장학금 Ⅱ유형 참여조건','◦(신·편입생지원조정)'),
 'work25':('moe-2025-scholarship-plan',5,'□ 근로장학생 지원 인원 확대','◦’가족돌봄청년'),
 'dream25':('moe-2025-scholarship-plan',6,'□ 드림장학생 선발 및 지원','4주거안정장학금'),
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/evaluation-v1')
    args=parser.parse_args()
    if args.output.exists(): parser.error('Frozen output exists; use a new version, never overwrite')
    snapshots={s['source_id']:s for s in read_jsonl(ROOT/'data/retrieval_snapshot.jsonl')}
    sources={s['source_id']:s for s in read_jsonl(ROOT/'data/sources.jsonl')}
    corpus=json.loads((ROOT/'config/corpus.json').read_text())['sources']
    training=read_jsonl(ROOT/'outputs/distillation-smoke-v1/examples.jsonl')
    trained_pages={(d['source_id'],d['pdf_page']) for row in training for d in row['documents']}
    trained_groups={row['leakage_group'] for row in training}
    docs={}
    checked=set()
    for key,(sid,num,begin,end) in SELECTORS.items():
        snapshot=snapshots[sid]
        if sid not in checked:
            if sha256_file(ROOT/snapshot['raw_path'])!=snapshot['sha256']: raise ValueError('Source changed')
            checked.add(sid)
        if (sid,num) in trained_pages: raise ValueError('Evaluation evidence page was in training')
        page=next(p for p in read_jsonl(ROOT/snapshot['extracted_path']) if p['pdf_page']==num)
        if page['sha256']!=snapshot['sha256']: raise ValueError('Extraction provenance differs')
        text=normalize(page['text']); start=text.index(begin); stop=text.index(end,start+len(begin))
        docs[key]={'id':f'{sid}:p{num}:{key}','source_id':sid,'year':sources[sid]['academic_year'],
                   'institution':corpus[sid]['institution'],'pdf_page':num,'sha256':snapshot['sha256'],
                   'text':text[start:stop].strip()}
    cases=seed_cases(); rng=random.Random(20261103)
    labels=list('ABCD')*(len(cases)//4); rng.shuffle(labels)
    index=read_jsonl(ROOT/'outputs/index.jsonl')
    test_rows=[]; inputs=[]
    for c,label in zip(cases,labels,strict=True):
        if c['leakage_group'] in trained_groups: raise ValueError('Training group overlap')
        if any(normalize(c['question'])==normalize(t['question']) for t in training): raise ValueError('Question overlap')
        correct=c.pop('correct_text'); alternatives=[a for a in c.pop('choices') if a!=correct]; rng.shuffle(alternatives)
        alternatives.insert('ABCD'.index(label),correct)
        context=[docs[k] for k in c.pop('evidence_keys')]; rng.shuffle(context)
        c.update(choices=dict(zip('ABCD',alternatives,strict=True)),correct=label,
                 expected_action=c.get('expected_action','answer'),
                 review_status='assistant_source_checked_not_independently_human_reviewed',
                 institution='부산대학교' if c['task']=='admissions' else '교육부',
                 evidence=[{k:d[k] for k in ['id','source_id','year','pdf_page','sha256']} for d in context])
        test_rows.append(c)
        base={k:c[k] for k in ['id','question','year','task','institution','choices']}
        base['oracle_documents']=context
        if c.get('free'):
            retrieved=retrieve(index,c['question'],year=c['year'],task=c['task'],institution=c['institution'],k=2) if c['year'] else []
            # Missing-year questions cannot safely retrieve a single-year policy.
            base['retrieved_documents']=[{**d,'id':d['chunk_id']} for d in retrieved]
        inputs.append(base)
    args.output.mkdir(parents=True)
    for name,rows in [('gold.jsonl',test_rows),('inputs.jsonl',inputs)]:
        (args.output/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    protocol={
      'version':'holdout-v1','frozen_at':datetime.now(timezone.utc).isoformat(),
      'mcq_questions':len(cases),'free_questions':sum(bool(c.get('free')) for c in cases),
      'rule_groups':len({c['leakage_group'] for c in cases}),'seed':20261103,
      'models':['base','sft','opsd'],'model_selection_after_test':False,
      'base_model':'Qwen3-4B-Q4_K_M.gguf','backend':'llama.cpp b11960; same base quantization for all arms',
      'oracle_mcq':'Explicit evidence snippets; four options; grammar restricts output to A/B/C/D; no reasoning or full-advice claim',
      'free_rag':'8 prespecified questions without options; BM25 k=2 frozen; schema NOT constrained; max_tokens=160; count malformed/truncated responses as failures',
      'generation':{'temperature':0,'seed':42,'enable_thinking':False,'cache_prompt':False},
      'mcq_scoring':'Exact selected-letter accuracy; paired discordances; rule-group bootstrap confidence intervals',
      'free_scoring':'Frozen required/forbidden patterns + JSON + relevant-year evidence ID; report structural and strict content separately; inspect errors without changing rubric',
      'independent_human_review':False,
      'training_qa_count':len(training),'overlap_training_pages':0,'overlap_training_rule_groups':0,
      'limitations':['One held-out admissions institution, plus national scholarship policy; not representative of Korea',
                    'Pretraining exposure cannot be excluded','Gold is assistant source-checked, not independent human adjudication',
                    'MCQ and free subset are different protocols, never pooled into one accuracy',
                    'BF16-trained adapters evaluated on the same quantized deployment base; not BF16 results'],
      'hashes':{name:sha256_file(args.output/name) for name in ['gold.jsonl','inputs.jsonl']},
      'input_index_sha256':sha256_file(ROOT/'outputs/index.jsonl'),
      'training_examples_sha256':sha256_file(ROOT/'outputs/distillation-smoke-v1/examples.jsonl'),
      'model_files':{str(p):sha256_file(ROOT/p) for p in ['models/Qwen3-4B-Q4_K_M.gguf','outputs/sft-adapter-f32.gguf','outputs/opsd-adapter-f32.gguf']}}
    write_json(args.output/'protocol.json',protocol)
    print(json.dumps({k:v for k,v in protocol.items() if k not in ['model_files','limitations']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
