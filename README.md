# 타이어 몰드 벤트 홀 검출 프로젝트

## 프로젝트 개요
이 저장소는 딥러닝 기반의 타이어 몰드 벤트 홀 자동 검출을 목표.
산업 환경 이미지에서 소형 객체 검출 성능(F1-score) 향상을 중점으로 개발

주요 기술:
- YOLOv8 기반 객체 검출
- SAHI(슬라이싱 추론)를 활용한 미세 홀 검출
- 조명 변화 대응을 위한 Adaptive CLAHE 전처리
- Albumentations 기반 데이터 증강

## 저장소 범위
이 저장소는 학습/추론을 위한 독립 실행 워크플로우만 포함합니다.
회사 내부 서버, IPC, 배포 관련 코드는 제외되어 있습니다.

## 요구 사항
- Python 3.8 이상
- CUDA 지원 GPU (학습 및 고속 추론 시 권장)

## 재현 가능한 실행 절차 (Windows PowerShell 기준)
1. 저장소 이동:
```bash
cd tire-hole-detection-project
```

2. 가상환경 생성 및 활성화:
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3. 의존성 설치:
```bash
pip install -r requirements.txt
```

4. `data.yaml` 수정 (로컬 데이터셋 경로 반영):
```yaml
path: ./dataset
train: images/train
val: images/val
nc: 1
names: ['vent_hole']
```

5. 데이터셋 최소 구조 확인:
```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

6. 학습 1 epoch 스모크 테스트(환경 검증용):
```bash
python training/train_yolov8.py --data data.yaml --epochs 1
```

7. 추론 스모크 테스트:
```bash
python inference/predict.py --model runs/detect/tire_hole_v8/weights/best.pt --image path/to/sample.jpg
```

## 프로젝트 구조
- `data_processing/`: 데이터 증강 및 YOLO 라벨 정규화 스크립트
- `training/`: YOLOv8 학습 진입 스크립트
- `inference/`: Adaptive CLAHE 전처리를 포함한 기본 추론 스크립트
- `tools/`: 보조 도구 스크립트(간단 라벨링 도우미, SAHI 테스트)

## 데이터셋 설정
학습 스크립트의 기본 데이터 설정 경로는 `data.yaml` 입니다.

`data.yaml` 내부의 데이터셋 경로를 로컬 환경에 맞게 수정합니다.

기본 예시는 아래와 같습니다:
```yaml
path: ./dataset
train: images/train
val: images/val

nc: 1
names: ['vent_hole']
```

## 사용 방법
1. YOLO 라벨 정규화 (선택):
   ```bash
   python data_processing/class_normalization.py --dir dataset/labels/train
   ```
2. 데이터 증강:
   ```bash
   python data_processing/augmentation.py
   ```
   `augmentation.py`는 기본 템플릿 스크립트입니다. 실행 전 파일 내부의
   `image_dir`, `label_dir`, `save_dir`를 환경에 맞게 수정하고,
   처리 루프를 추가해 사용하세요.
3. 모델 학습:
   ```bash
   python training/train_yolov8.py --data data.yaml --epochs 50
   ```
4. 기본 추론:
   ```bash
   python inference/predict.py --model path/to/best.pt --image path/to/sample.jpg
   ```
5. 슬라이싱 추론(SAHI):
   ```bash
   python tools/test_sahi.py --model path/to/best.pt --image path/to/sample.jpg
   ```

## 참고 사항
- `runs/`와 같은 학습 결과 폴더는 기본적으로 Git 추적에서 제외됩니다.
- 모델 가중치(`*.pt`, `*.pth`)와 데이터셋은 기본적으로 Git 추적에서 제외됩니다.
- `tools/labeling_tool.py`는 데모용 경량 도구이며, 실무 라벨링에는 전용 라벨링 툴 사용을 권장합니다.
