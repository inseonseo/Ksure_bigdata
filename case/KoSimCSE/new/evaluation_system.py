"""
보험사고 유사도 시스템 성능 평가를 위한 Train/Valid/Test 분할 및 평가 시스템
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import os
from datetime import datetime

class InsuranceEvaluationSystem:
    def __init__(self, data_path, preserve_labels=True, min_support_for_test=2):
        """
        초기화
        
        Args:
            data_path: CSV 데이터 파일 경로
            preserve_labels: True이면 라벨(판정구분/판정사유)을 어떤 방식으로도 통합/변환하지 않음
            min_support_for_test: 테스트에 포함시키기 위한 클래스 최소 건수(미만은 test에서 제외)
        """
        self.data_path = data_path
        self.df = None
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        self.label_encoders = {}
        
        # 추가 설정
        self.preserve_labels = preserve_labels
        self.min_support_for_test = min_support_for_test
        self.excluded_classes_from_test = []
        
    def load_and_prepare_data(self):
        """데이터 로드 및 전처리"""
        print("📊 데이터 로딩 중...")
        
        # 데이터 로드
        self.df = pd.read_csv(self.data_path, encoding='cp949')
        print(f"전체 데이터: {len(self.df):,}건")
        
        # 기본 전처리
        self.df = self.df.dropna(subset=['판정구분', '판정사유'])
        print(f"유효한 판정 데이터: {len(self.df):,}건")
        
        if not self.preserve_labels:
            # 라벨 통합이 허용된 경우에만 수행
            self.df = self._consolidate_judgment_categories()
            self.df = self._consolidate_reason_categories()
        else:
            print("🔒 라벨 보존 모드: 판정구분/판정사유를 그대로 사용합니다.")
        
        # 분포 확인
        print("\n📈 판정구분 분포:")
        print(self.df['판정구분'].value_counts())
        print("\n📈 판정사유 분포 (상위 10개):")
        print(self.df['판정사유'].value_counts().head(10))
        
        return self.df
    
    def _consolidate_judgment_categories(self):
        """판정구분 카테고리 통합 (최소한의 통합만)"""
        print("\n🔄 판정구분 카테고리 정리 중...")
        
        # 원본 분포
        original_counts = self.df['판정구분'].value_counts()
        print("원본 분포:")
        print(original_counts)
        
        # 최소한의 통합만 수행 (업무적으로 의미있는 통합)
        judgment_mapping = {
            '지급': '지급',
            '면책': '면책',
            '지급유예': '지급유예',
            '가지급': '지급',  # 가지급은 지급의 일종
            '기타지급거절': '지급거절',  # 지급거절로 명확화
            '보험관계불성립': '면책'  # 보험관계불성립은 면책의 일종
        }
        
        # 매핑 적용
        self.df['판정구분_original'] = self.df['판정구분'].copy()  # 원본 보존
        self.df['판정구분'] = self.df['판정구분'].map(judgment_mapping).fillna('기타')
        
        # 통합 후 분포
        consolidated_counts = self.df['판정구분'].value_counts()
        print("\n정리 후 분포:")
        print(consolidated_counts)
        
        # 통합 통계
        total_consolidated = len(self.df)
        print(f"\n📊 정리 결과:")
        for category, count in consolidated_counts.items():
            percentage = (count / total_consolidated) * 100
            print(f"   - {category}: {count:,}건 ({percentage:.1f}%)")
        
        return self.df
    
    def _consolidate_reason_categories(self):
        """판정사유 카테고리 정리 (최소한의 통합만)"""
        print("\n🔄 판정사유 카테고리 정리 중...")
        
        # 원본 분포 (상위 20개)
        original_counts = self.df['판정사유'].value_counts()
        print("원본 분포 (상위 20개):")
        print(original_counts.head(20))
        
        # 최소한의 통합만 수행 (의미적으로 동일한 것들만)
        reason_mapping = {
            # 지급 관련
            '지급 판정': '지급판정',
            '기타 지급사유의 해소': '기타지급사유해소',
            
            # 면책 관련 - 주요 사유들
            '보험계약자의 고의 또는 과실로 인하여 발생한 손실': '고의과실',
            '연속수출': '연속수출',
            '보상한도를 초과하는 손실': '보상한도초과',
            '보험관계의 성립': '보험관계성립',
            '주의의무 해태로 인한 손실가중': '주의의무해태',
            '신용보증조건 위반': '신용보증조건위반',
            '신용보증관계의 성립': '신용보증관계성립',
            '권리보전의무 해태로 인한 손실가중': '권리보전의무해태',
            '보험계약의 해지(고지의무, 내용변경, 보험료 미납)': '보험계약해지',
            '보험계약이 공사가 책임 지울 수 없는 사유로 무효 실효 해제 해지': '보험계약무효',
            '보험증권 상 특약사항 해태로 인한 손실가중': '특약사항해태',
            
            # 지급유예 관련
            '지급유예 판정': '지급유예판정',
            '지급할 보험금을 산정하기 위하여 장기간이 소요되는 경우': '장기간소요',
            '사고원인의 조사에 장기간이 소요되는 경우': '사고원인조사장기간',
            
            # 기타 주요 사유들
            '적용대상 수출거래': '적용대상수출거래',
            '변제충당': '변제충당',
            '보험계약자와 수출계약상대방간에 분쟁발생': '분쟁발생',
            '보증채무의 범위': '보증채무범위',
            '사고발생통지 의무': '사고발생통지의무',
            '신용보증대상 수출거래': '신용보증대상수출거래',
            '손실방지 경감의무': '손실방지경감의무',
            '물품의 멸실, 훼손 또는 기타 물품에 대해 발생한 손실': '물품손실',
            '본지사거래에서 신용위험으로 인하여 발생한 손실': '본지사거래손실',
            '무신용장방식 거래에서 수출계약의 주요 사항 위반': '수출계약위반',
            '보험책임 개시일전에 발생한 손실': '책임개시일전손실',
            '금융계약상의 의무사항 불이행': '금융계약의무불이행',
            '지시에 따를 의무': '지시따를의무',
            '수출채권 감소(상계, 채무면제 등)': '수출채권감소',
            '무신용장방식거래에서 수입자의 신용상태악화를 인지한 이후 수출거래': '수입자신용악화',
            '조사에 따를 의무': '조사따를의무',
            '상계처리': '상계처리',
            '사유 없음': '사유없음',
            '법령을 위반하여 취득한 채권': '법령위반채권',
            '책임금액을 초과하는 손실': '책임금액초과',
            '보험사고 관련 수출물품이 처분되지 않는 경우': '수출물품미처분',
            '신용보증부대출의 실행금지': '신용보증부대출실행금지',
            '신용장방식 등의 수출거래의 정의에 위배되는 신용장거래': '신용장거래위배',
            '신용장방식 거래에서 신용장조건 위반': '신용장조건위반',
            '신용보증부대출에의 우선충당': '신용보증부대출우선충당',
            '적용대상거래 및 보험계약의 성립(부보대상거래, 조건부 신용장, 대금지급책임 면제 등)': '적용대상거래성립',
            '보험계약자와 재판매계약상대방간에 분쟁발생': '재판매계약분쟁'
        }
        
        # 매핑 적용
        self.df['판정사유_original'] = self.df['판정사유'].copy()  # 원본 보존
        self.df['판정사유'] = self.df['판정사유'].map(reason_mapping).fillna('기타사유')
        
        # 정리 후 분포 (상위 15개)
        consolidated_counts = self.df['판정사유'].value_counts()
        print("\n정리 후 분포 (상위 15개):")
        print(consolidated_counts.head(15))
        
        # 정리 통계
        total_consolidated = len(self.df)
        print(f"\n📊 판정사유 정리 결과:")
        for category, count in consolidated_counts.head(10).items():
            percentage = (count / total_consolidated) * 100
            print(f"   - {category}: {count:,}건 ({percentage:.1f}%)")
        
        return self.df
    
    def create_train_valid_test_split(self, test_size=0.2, valid_size=0.1, random_state=42):
        """
        데이터를 Train/Valid/Test로 분할 (개선된 층화 샘플링)
        """
        print(f"\n🔄 데이터 분할 중... (Train: {100-test_size*100-valid_size*100:.0f}% / Valid: {valid_size*100:.0f}% / Test: {test_size*100:.0f}%)")
        
        # 판정구분 기준 층화 타겟 생성
        self.df = self._create_balanced_stratification_target()
        
        # 희소 클래스(판정구분) 처리: 테스트에서 제외하고 학습에는 포함
        counts = self.df['판정구분'].value_counts()
        rare_classes = counts[counts < self.min_support_for_test].index.tolist()
        self.excluded_classes_from_test = rare_classes
        if rare_classes:
            print(f"⚠️ 희소 클래스(테스트 제외): {rare_classes}  (min_support_for_test={self.min_support_for_test})")
        
        base_df = self.df[~self.df['판정구분'].isin(rare_classes)].copy()
        rare_df = self.df[self.df['판정구분'].isin(rare_classes)].copy()
        
        # base_df만 stratify로 분할
        train_base, temp = train_test_split(
            base_df,
            test_size=(test_size + valid_size),
            random_state=random_state,
            stratify=base_df['stratify_target']
        )
        test_ratio = test_size / (test_size + valid_size)
        valid_base, test_base = train_test_split(
            temp,
            test_size=test_ratio,
            random_state=random_state,
            stratify=temp['stratify_target']
        )
        
        # 희소 클래스는 train으로만 넣음(또는 필요 시 valid에 일부 분배 가능)
        self.train_df = pd.concat([train_base, rare_df], ignore_index=True)
        self.valid_df = valid_base.copy()
        self.test_df = test_base.copy()  # test에는 희소 클래스 없음
        
        # 결과 출력
        print(f"✅ 분할 완료:")
        print(f"   - Train: {len(self.train_df):,}건 ({len(self.train_df)/len(self.df)*100:.1f}%)")
        print(f"   - Valid: {len(self.valid_df):,}건 ({len(self.valid_df)/len(self.df)*100:.1f}%)")
        print(f"   - Test:  {len(self.test_df):,}건 ({len(self.test_df)/len(self.df)*100:.1f}%)")
        
        print("\n📊 세트별 판정구분 분포:")
        for name, dataset in [('Train', self.train_df), ('Valid', self.valid_df), ('Test', self.test_df)]:
            dist = dataset['판정구분'].value_counts(normalize=True)
            print(f"{name}: {dict(dist.round(3))}")
        
        if self.excluded_classes_from_test:
            print(f"\n📝 테스트 제외 클래스(학습에는 포함): {self.excluded_classes_from_test}")
        
        return self.train_df, self.valid_df, self.test_df
    
    def _create_balanced_stratification_target(self):
        """균형잡힌 층화 샘플링을 위한 타겟 변수 생성"""
        print("\n🎯 균형잡힌 층화 샘플링 타겟 생성 중...")
        
        # 1. 판정구분별 분포 확인
        judgment_counts = self.df['판정구분'].value_counts()
        print("판정구분별 분포:")
        print(judgment_counts)
        
        # 2. 주요 판정사유별 분포 확인 (상위 10개)
        reason_counts = self.df['판정사유'].value_counts()
        print("\n주요 판정사유별 분포 (상위 10개):")
        print(reason_counts.head(10))
        
        # 3. 균형잡힌 층화 타겟 생성 (판정구분 기반으로 단순화)
        # 판정구분이 업무적으로 더 중요하므로 이를 기준으로 층화
        self.df['stratify_target'] = self.df['판정구분']
        
        # 4. 층화 타겟 분포 확인
        final_counts = self.df['stratify_target'].value_counts()
        print(f"\n📊 층화 타겟 분포 (판정구분 기반):")
        for category, count in final_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"   - {category}: {count:,}건 ({percentage:.1f}%)")
        
        # 5. 균형도 평가
        min_count = final_counts.min()
        max_count = final_counts.max()
        balance_ratio = min_count / max_count if max_count > 0 else 0
        print(f"\n✅ 층화 타겟 균형도: {balance_ratio:.3f} (최소: {min_count}, 최대: {max_count})")
        
        return self.df
    
    def prepare_features_for_modeling(self):
        """모델링을 위한 피처 엔지니어링"""
        print("\n🛠️ 피처 엔지니어링 중...")
        
        # 범주형 변수 인코딩
        categorical_features = ['수입국', '사고유형명', '보험종목', '상품분류명', '결제방법']
        
        # Train 데이터로 Label Encoder 학습
        for feature in categorical_features:
            if feature in self.train_df.columns:
                le = LabelEncoder()
                
                # 결측값 처리
                train_values = self.train_df[feature].fillna('Unknown').astype(str)
                le.fit(train_values)
                self.label_encoders[feature] = le
                
                # 모든 데이터셋에 적용 (미지 카테고리는 'Unknown' 코드로 대체)
                for df_name, df in [('train', self.train_df), ('valid', self.valid_df), ('test', self.test_df)]:
                    values = df[feature].fillna('Unknown').astype(str)

                    # LabelEncoder에 'Unknown' 클래스가 없으면 추가 (정렬 보장)
                    if 'Unknown' not in le.classes_:
                        le.classes_ = np.sort(np.append(le.classes_, 'Unknown'))

                    # 폴백 코드(Unknown)
                    unknown_code = le.transform(['Unknown'])[0]

                    # 알려진 값 마스크
                    known_mask = values.isin(le.classes_)

                    # 전부 Unknown 코드로 채운 후, 알려진 값만 변환해서 덮어씀
                    encoded_values = np.full(len(values), unknown_code)
                    if known_mask.any():
                        encoded_values[known_mask] = le.transform(values[known_mask])

                    df[f'{feature}_encoded'] = encoded_values
                    
                print(f"   - {feature}: {len(le.classes_)}개 카테고리")
        
        # 숫자형 피처 정규화
        numeric_features = ['원화사고금액', '부보율']
        for feature in numeric_features:
            if feature in self.train_df.columns:
                # Train 데이터로 정규화 파라미터 계산
                train_mean = self.train_df[feature].mean()
                train_std = self.train_df[feature].std()

                # 분산 0 방어 (정규화 불가 시 0으로 설정)
                if pd.isna(train_std) or train_std == 0:
                    for df_name, df in [('train', self.train_df), ('valid', self.valid_df), ('test', self.test_df)]:
                        df[f'{feature}_normalized'] = 0.0
                    print(f"   - {feature}: 표준편차 0 → 정규화 생략(0으로 설정)")
                    continue

                # 모든 데이터셋에 적용
                for df_name, df in [('train', self.train_df), ('valid', self.valid_df), ('test', self.test_df)]:
                    df[f'{feature}_normalized'] = (df[feature] - train_mean) / train_std

                print(f"   - {feature}: 정규화 완료 (평균: {train_mean:.2f}, 표준편차: {train_std:.2f})")
    
    def evaluate_similarity_system(self, similarity_system, sample_size=300):
        """
        유사도 시스템 성능 평가 (다중 타겟 평가)
        
        Args:
            similarity_system: 평가할 유사도 시스템 인스턴스
            sample_size: 평가에 사용할 샘플 수
        """
        print(f"\n🔍 유사도 시스템 성능 평가 (샘플 크기: {sample_size})")
        
        # 테스트 데이터에서 층화 샘플링으로 샘플 추출
        test_sample = self._get_stratified_test_sample(sample_size)
        
        # 다중 타겟 평가를 위한 결과 저장
        results = {
            'judgment': {'predictions': [], 'actuals': [], 'similarity_scores': [], 'confidence_scores': []},
            'reason': {'predictions': [], 'actuals': [], 'similarity_scores': [], 'confidence_scores': []}
        }
        
        for idx, test_case in test_sample.iterrows():
            try:
                # 테스트 케이스를 쿼리로 변환
                case_data = {
                    '수입국': test_case['수입국'],
                    '보험종목': test_case['보험종목'],
                    '사고유형명': test_case['사고유형명'],
                    '원화사고금액': test_case['원화사고금액'],
                    '사고설명': test_case['사고설명']
                }
                
                # Train 데이터에서 유사사례 검색 (자기 자신 제외)
                search_df = self.train_df[self.train_df.index != idx].copy()
                
                # 유사도 계산
                similarities = similarity_system.calculate_similarity_scores(case_data, search_df.head(1000))
                
                if similarities:
                    # 상위 5개 유사사례 분석
                    top_5 = similarities[:5]

                    # 가중 다수결: 유사도 점수를 가중치로 합산하여 최다 가중 클래스를 예측
                    def weighted_vote(label_list, weights):
                        scores = {}
                        for lbl, w in zip(label_list, weights):
                            scores[lbl] = scores.get(lbl, 0.0) + float(w)
                        # 최대 가중치 라벨 반환
                        return max(scores.items(), key=lambda x: x[1])[0], scores

                    top_scores = [case[0] for case in top_5]
                    judgment_labels = [case[3]['판정구분'] for case in top_5]
                    reason_labels = [case[3]['판정사유'] for case in top_5]

                    pred_judgment, judgment_score_map = weighted_vote(judgment_labels, top_scores)
                    pred_reason, reason_score_map = weighted_vote(reason_labels, top_scores)

                    # 면책 오버라이드: 최상위 1건이 면책이고 종합유사도 임계치 이상이면 면책으로 고정
                    top1_score, _, _, top1_case = top_5[0]
                    if top1_case['판정구분'] == '면책' and top1_score >= 0.65:
                        pred_judgment = '면책'

                    # 신뢰도: 가중 합 중 예측 라벨 비율
                    sum_w = sum(top_scores) if top_scores else 1.0
                    judgment_confidence = (judgment_score_map.get(pred_judgment, 0.0) / sum_w) if sum_w else 0.0
                    reason_confidence = (reason_score_map.get(pred_reason, 0.0) / sum_w) if sum_w else 0.0

                    # 평균 유사도
                    avg_similarity = np.mean(top_scores) if top_scores else 0.0
                    
                    # 결과 저장
                    results['judgment']['predictions'].append(pred_judgment)
                    results['judgment']['actuals'].append(test_case['판정구분'])
                    results['judgment']['similarity_scores'].append(avg_similarity)
                    results['judgment']['confidence_scores'].append(judgment_confidence)
                    
                    results['reason']['predictions'].append(pred_reason)
                    results['reason']['actuals'].append(test_case['판정사유'])
                    results['reason']['similarity_scores'].append(avg_similarity)
                    results['reason']['confidence_scores'].append(reason_confidence)
                    
            except Exception as e:
                print(f"   ⚠️ 평가 중 오류 (인덱스 {idx}): {e}")
                continue
        
        # 성능 지표 계산
        if results['judgment']['predictions'] and results['reason']['predictions']:
            final_results = self._calculate_multi_target_metrics(results)
            self._print_multi_target_results(final_results)
            return final_results
        else:
            print("❌ 평가할 데이터가 없습니다.")
            return None
    
    def _calculate_multi_target_metrics(self, results):
        """다중 타겟 성능 지표 계산"""
        from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score
        from sklearn.metrics import precision_recall_fscore_support
        
        final_results = {}
        
        for target_type in ['judgment', 'reason']:
            predictions = results[target_type]['predictions']
            actuals = results[target_type]['actuals']
            similarity_scores = results[target_type]['similarity_scores']
            confidence_scores = results[target_type]['confidence_scores']
            
            # 기본 정확도
            accuracy = sum(p == a for p, a in zip(predictions, actuals)) / len(predictions)
            
            # 균형잡힌 정확도
            balanced_acc = balanced_accuracy_score(actuals, predictions)
            
            # F1 스코어
            f1_macro = f1_score(actuals, predictions, average='macro', zero_division=0)
            f1_micro = f1_score(actuals, predictions, average='micro', zero_division=0)
            
            # 혼동 행렬
            unique_labels = sorted(list(set(actuals + predictions)))
            cm = confusion_matrix(actuals, predictions, labels=unique_labels)
            
            # 분류 리포트
            report = classification_report(actuals, predictions, target_names=unique_labels, zero_division=0, output_dict=True)
            
            # 클래스별 성능
            class_performance = {}
            for label in unique_labels:
                if label in report:
                    class_performance[label] = {
                        'precision': report[label]['precision'],
                        'recall': report[label]['recall'],
                        'f1-score': report[label]['f1-score'],
                        'support': report[label]['support']
                    }
            
            # 통계
            similarity_stats = {
                'mean': np.mean(similarity_scores),
                'std': np.std(similarity_scores),
                'min': np.min(similarity_scores),
                'max': np.max(similarity_scores)
            }
            
            confidence_stats = {
                'mean': np.mean(confidence_scores),
                'std': np.std(confidence_scores),
                'min': np.min(confidence_scores),
                'max': np.max(confidence_scores)
            }

            # 면책 중심 바이너리 지표 (판정구분일 때만 계산)
            exemption_metrics = None
            if target_type == 'judgment' and predictions and actuals:
                y_true = [1 if a == '면책' else 0 for a in actuals]
                y_pred = [1 if p == '면책' else 0 for p in predictions]
                prec, rec, f1_bin, support_pos = precision_recall_fscore_support(
                    y_true, y_pred, average='binary', zero_division=0
                )
                exemption_metrics = {
                    'precision': float(prec),
                    'recall': float(rec),
                    'f1': float(f1_bin),
                    'positive_support': int(sum(y_true))
                }
            
            final_results[target_type] = {
                'accuracy': accuracy,
                'balanced_accuracy': balanced_acc,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'predictions': predictions,
                'actuals': actuals,
                'similarity_scores': similarity_scores,
                'confidence_scores': confidence_scores,
                'confusion_matrix': cm,
                'classification_report': report,
                'class_performance': class_performance,
                'similarity_stats': similarity_stats,
                'confidence_stats': confidence_stats,
                'exemption_metrics': exemption_metrics,
                'sample_size': len(predictions)
            }
        
        return final_results
    
    def _print_multi_target_results(self, results):
        """다중 타겟 평가 결과 출력"""
        print(f"\n📈 다중 타겟 성능 결과:")
        print("=" * 60)
        
        # 1. 판정구분 평가 결과
        print(f"\n🎯 판정구분 예측 성능:")
        judgment = results['judgment']
        print(f"   - 정확도: {judgment['accuracy']:.3f} ({judgment['accuracy']*100:.1f}%)")
        print(f"   - 균형잡힌 정확도: {judgment['balanced_accuracy']:.3f} ({judgment['balanced_accuracy']*100:.1f}%)")
        print(f"   - F1 스코어 (Macro): {judgment['f1_macro']:.3f}")
        print(f"   - F1 스코어 (Micro): {judgment['f1_micro']:.3f}")
        print(f"   - 평가 샘플 수: {judgment['sample_size']}개")
        print(f"   - 평균 유사도: {judgment['similarity_stats']['mean']:.3f} ± {judgment['similarity_stats']['std']:.3f}")
        print(f"   - 평균 신뢰도: {judgment['confidence_stats']['mean']:.3f} ± {judgment['confidence_stats']['std']:.3f}")
        if judgment.get('exemption_metrics'):
            em = judgment['exemption_metrics']
            print(f"   - 면책 탐지(이진) Precision: {em['precision']:.3f}, Recall: {em['recall']:.3f}, F1: {em['f1']:.3f} (면책 수: {em['positive_support']})")
        
        # 판정구분 클래스별 성능
        print(f"\n📊 판정구분 클래스별 성능:")
        for label, perf in judgment['class_performance'].items():
            print(f"   - {label}:")
            print(f"     • Precision: {perf['precision']:.3f}")
            print(f"     • Recall: {perf['recall']:.3f}")
            print(f"     • F1-Score: {perf['f1-score']:.3f}")
            print(f"     • Support: {perf['support']}건")
        
        # 2. 판정사유 평가 결과
        print(f"\n📋 판정사유 예측 성능:")
        reason = results['reason']
        print(f"   - 정확도: {reason['accuracy']:.3f} ({reason['accuracy']*100:.1f}%)")
        print(f"   - 균형잡힌 정확도: {reason['balanced_accuracy']:.3f} ({reason['balanced_accuracy']*100:.1f}%)")
        print(f"   - F1 스코어 (Macro): {reason['f1_macro']:.3f}")
        print(f"   - F1 스코어 (Micro): {reason['f1_micro']:.3f}")
        print(f"   - 평가 샘플 수: {reason['sample_size']}개")
        print(f"   - 평균 유사도: {reason['similarity_stats']['mean']:.3f} ± {reason['similarity_stats']['std']:.3f}")
        print(f"   - 평균 신뢰도: {reason['confidence_stats']['mean']:.3f} ± {reason['confidence_stats']['std']:.3f}")
        
        # 판정사유 클래스별 성능 (상위 10개만)
        print(f"\n📊 판정사유 클래스별 성능 (상위 10개):")
        sorted_reasons = sorted(reason['class_performance'].items(), 
                              key=lambda x: x[1]['support'], reverse=True)[:10]
        for label, perf in sorted_reasons:
            print(f"   - {label}:")
            print(f"     • Precision: {perf['precision']:.3f}")
            print(f"     • Recall: {perf['recall']:.3f}")
            print(f"     • F1-Score: {perf['f1-score']:.3f}")
            print(f"     • Support: {perf['support']}건")
        
        # 3. 종합 평가
        print(f"\n🎯 종합 평가:")
        print(f"   - 판정구분 예측 정확도: {judgment['accuracy']:.3f}")
        print(f"   - 판정사유 예측 정확도: {reason['accuracy']:.3f}")
        print(f"   - 평균 정확도: {(judgment['accuracy'] + reason['accuracy']) / 2:.3f}")
        
        # 혼동 행렬 (판정구분만 표시 - 판정사유는 너무 많음)
        print(f"\n📊 판정구분 혼동 행렬:")
        unique_labels = sorted(list(set(judgment['actuals'] + judgment['predictions'])))
        cm_df = pd.DataFrame(judgment['confusion_matrix'], index=unique_labels, columns=unique_labels)
        print(cm_df)
    
    def _get_stratified_test_sample(self, sample_size):
        """층화 샘플링으로 테스트 샘플 추출"""
        print(f"📊 층화 샘플링으로 {sample_size}개 샘플 추출 중...")
        
        # 판정구분별 분포 확인
        judgment_counts = self.test_df['판정구분'].value_counts()
        print("테스트 데이터 판정구분 분포:")
        print(judgment_counts)
        
        # 층화 샘플링으로 균형잡힌 샘플 추출
        stratified_sample = []
        
        for judgment in judgment_counts.index:
            judgment_data = self.test_df[self.test_df['판정구분'] == judgment]
            judgment_count = len(judgment_data)
            
            # 각 판정구분별로 비례하여 샘플 추출
            if judgment_count > 0:
                # 최소 5개, 최대 전체의 50%까지 추출
                min_samples = min(5, judgment_count)
                max_samples = min(int(judgment_count * 0.5), judgment_count)
                target_samples = min(max_samples, max(min_samples, int(sample_size * judgment_count / len(self.test_df))))
                
                if target_samples > 0:
                    sampled = judgment_data.sample(n=target_samples, random_state=42)
                    stratified_sample.append(sampled)
                    print(f"   - {judgment}: {target_samples}개 추출 (전체: {judgment_count}개)")
        
        if stratified_sample:
            final_sample = pd.concat(stratified_sample, ignore_index=True)
            print(f"✅ 총 {len(final_sample)}개 샘플 추출 완료")
            return final_sample
        else:
            print("⚠️ 층화 샘플링 실패, 랜덤 샘플링으로 대체")
            return self.test_df.sample(n=min(sample_size, len(self.test_df)), random_state=42)
    
    def _calculate_comprehensive_metrics(self, predictions, actuals, similarity_scores, confidence_scores):
        """포괄적인 성능 지표 계산"""
        from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, f1_score
        
        # 기본 정확도
        accuracy = sum(p == a for p, a in zip(predictions, actuals)) / len(predictions)
        
        # 균형잡힌 정확도 (불균형 데이터에 적합)
        balanced_acc = balanced_accuracy_score(actuals, predictions)
        
        # F1 스코어 (마크로 평균)
        f1_macro = f1_score(actuals, predictions, average='macro', zero_division=0)
        
        # F1 스코어 (마이크로 평균)
        f1_micro = f1_score(actuals, predictions, average='micro', zero_division=0)
        
        # 혼동 행렬
        unique_labels = sorted(list(set(actuals + predictions)))
        cm = confusion_matrix(actuals, predictions, labels=unique_labels)
        
        # 분류 리포트
        report = classification_report(actuals, predictions, target_names=unique_labels, zero_division=0, output_dict=True)
        
        # 클래스별 성능
        class_performance = {}
        for label in unique_labels:
            if label in report:
                class_performance[label] = {
                    'precision': report[label]['precision'],
                    'recall': report[label]['recall'],
                    'f1-score': report[label]['f1-score'],
                    'support': report[label]['support']
                }
        
        # 유사도 및 신뢰도 통계
        similarity_stats = {
            'mean': np.mean(similarity_scores),
            'std': np.std(similarity_scores),
            'min': np.min(similarity_scores),
            'max': np.max(similarity_scores)
        }
        
        confidence_stats = {
            'mean': np.mean(confidence_scores),
            'std': np.std(confidence_scores),
            'min': np.min(confidence_scores),
            'max': np.max(confidence_scores)
        }
        
        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'predictions': predictions,
            'actuals': actuals,
            'similarity_scores': similarity_scores,
            'confidence_scores': confidence_scores,
            'confusion_matrix': cm,
            'classification_report': report,
            'class_performance': class_performance,
            'similarity_stats': similarity_stats,
            'confidence_stats': confidence_stats,
            'sample_size': len(predictions)
        }
    
    def _print_evaluation_results(self, results):
        """평가 결과 출력"""
        print(f"\n📈 성능 결과:")
        print(f"   - 정확도: {results['accuracy']:.3f} ({results['accuracy']*100:.1f}%)")
        print(f"   - 균형잡힌 정확도: {results['balanced_accuracy']:.3f} ({results['balanced_accuracy']*100:.1f}%)")
        print(f"   - F1 스코어 (Macro): {results['f1_macro']:.3f}")
        print(f"   - F1 스코어 (Micro): {results['f1_micro']:.3f}")
        print(f"   - 평가 샘플 수: {results['sample_size']}개")
        print(f"   - 평균 유사도: {results['similarity_stats']['mean']:.3f} ± {results['similarity_stats']['std']:.3f}")
        print(f"   - 평균 신뢰도: {results['confidence_stats']['mean']:.3f} ± {results['confidence_stats']['std']:.3f}")
        
        # 클래스별 성능
        print(f"\n📊 클래스별 성능:")
        for label, perf in results['class_performance'].items():
            print(f"   - {label}:")
            print(f"     • Precision: {perf['precision']:.3f}")
            print(f"     • Recall: {perf['recall']:.3f}")
            print(f"     • F1-Score: {perf['f1-score']:.3f}")
            print(f"     • Support: {perf['support']}건")
        
        # 혼동 행렬
        print(f"\n📊 혼동 행렬:")
        unique_labels = sorted(list(set(results['actuals'] + results['predictions'])))
        cm_df = pd.DataFrame(results['confusion_matrix'], index=unique_labels, columns=unique_labels)
        print(cm_df)
        
        # 분류 리포트
        print(f"\n📋 상세 분류 리포트:")
        report_df = pd.DataFrame(results['classification_report']).transpose()
        print(report_df.round(3))
    
    def save_splits(self, save_dir='evaluation_data'):
        """분할된 데이터셋 저장"""
        print(f"\n💾 데이터셋 저장 중... ({save_dir})")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 데이터셋 저장
        self.train_df.to_csv(f'{save_dir}/train_data.csv', index=False, encoding='utf-8-sig')
        self.valid_df.to_csv(f'{save_dir}/valid_data.csv', index=False, encoding='utf-8-sig')
        self.test_df.to_csv(f'{save_dir}/test_data.csv', index=False, encoding='utf-8-sig')
        
        # Label Encoder 저장
        with open(f'{save_dir}/label_encoders.pkl', 'wb') as f:
            pickle.dump(self.label_encoders, f)
        
        # 메타데이터 저장
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'total_samples': len(self.df),
            'train_samples': len(self.train_df),
            'valid_samples': len(self.valid_df),
            'test_samples': len(self.test_df),
            'features': list(self.label_encoders.keys())
        }
        
        with open(f'{save_dir}/metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ 저장 완료:")
        print(f"   - Train: {save_dir}/train_data.csv")
        print(f"   - Valid: {save_dir}/valid_data.csv")
        print(f"   - Test: {save_dir}/test_data.csv")
        print(f"   - Encoders: {save_dir}/label_encoders.pkl")
        print(f"   - Metadata: {save_dir}/metadata.pkl")

def main():
    """메인 실행 함수"""
    print("🚀 보험사고 유사도 시스템 성능 평가 시작")
    
    # 평가 시스템 초기화
    eval_system = InsuranceEvaluationSystem('data/design.csv')
    
    # 데이터 로드
    eval_system.load_and_prepare_data()
    
    # Train/Valid/Test 분할
    eval_system.create_train_valid_test_split()
    
    # 피처 엔지니어링
    eval_system.prepare_features_for_modeling()
    
    # 분할된 데이터 저장
    eval_system.save_splits()
    
    print("\n✅ 평가 시스템 준비 완료!")
    print("💡 이제 다음과 같이 사용할 수 있습니다:")
    print("   1. train_data.csv로 모델 학습")
    print("   2. valid_data.csv로 하이퍼파라미터 튜닝")
    print("   3. test_data.csv로 최종 성능 평가")
    print("\n🔧 유사도 시스템 평가 예시:")
    print("   eval_result = eval_system.evaluate_similarity_system(your_similarity_system)")

if __name__ == "__main__":
    main()