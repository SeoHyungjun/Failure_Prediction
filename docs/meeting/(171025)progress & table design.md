# progress
1. 환경구축
normal 데이터 수집 시작

* ceph image 사용하는 VM 생성
  - 일반 RBD 만들어서 VM host에 마운트, 해당 디렉토리에 VM 이미지 생성 후 가상 머신 생성

* base table 생성을 위한 자료조사
  - 테이블 초기화(완료)
  - 예측 알고리즘의 기본 input 형태 가공
     => 테이블에 삭제 및 추가(완료). SQL

* 지속적 I/O 발생
  - filebench : fileserver, mailserver, webserver
<br/>
2. 자동화(형태만)
  - 장애 체크
  - 장애 발생기
<br/>
3. 프레임워크 설계 X
<br/>

# base table
* 테이블 생성
  - 클라이언트(1,2)
  - 클러스터(1,2,3) 
  - 시스템 상태
<br/>
* 테이블 내용
  * 1. 클라이언트
    - 시스템 메트릭
    - 노드 장애 발생 여부

  * 2. 클러스터
    - 시스템 + ceph 메트릭
    - 노드 장애 발생 여부

  * 3. 시스템 상태(pool 단위 장애, 재점검 필요!)
    - 시간, 서비스, ceph health, 각 노드별 상태(비트맵), I/O possible??
