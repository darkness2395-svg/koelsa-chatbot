import streamlit as st
import os
import re
import tempfile
from pathlib import Path

st.set_page_config(page_title="경남동부지사 톡톡AI", page_icon="🛗", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
* { font-family: 'Noto Sans KR', sans-serif; }
.main { background-color: #f0f4f8; }
.header-box {
    background: linear-gradient(135deg, #1a3a6b 0%, #2563eb 100%);
    border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;
    color: white; display: flex; align-items: center; gap: 16px;
}
.header-box h1 { margin: 0; font-size: 1.7rem; font-weight: 700; }
.header-box p  { margin: 4px 0 0; font-size: 0.9rem; opacity: 0.85; }
.msg-user {
    background: #2563eb; color: white;
    border-radius: 18px 18px 4px 18px; padding: 12px 18px;
    margin: 8px 0 8px auto; max-width: 75%; width: fit-content;
    margin-left: auto; font-size: 0.95rem; line-height: 1.6;
}
.msg-bot {
    background: white; color: #1e293b;
    border-radius: 18px 18px 18px 4px; padding: 14px 18px;
    margin: 8px auto 8px 0; max-width: 80%; width: fit-content;
    font-size: 0.95rem; line-height: 1.7;
    border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.source-tag {
    display: inline-block; background: #eff6ff; color: #2563eb;
    font-size: 0.75rem; border-radius: 6px; padding: 2px 8px;
    margin: 8px 4px 0 0; border: 1px solid #bfdbfe;
}
.status-ok   { color: #16a34a; font-weight: 600; font-size: 0.85rem; }
.status-warn { color: #d97706; font-weight: 600; font-size: 0.85rem; }
.stButton > button {
    border-radius: 10px; border: 1px solid #cbd5e1;
    background: white; color: #334155; font-size: 0.83rem; padding: 6px 12px;
}
.stButton > button:hover { background: #eff6ff; border-color: #93c5fd; color: #1d4ed8; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <div style="font-size:2.8rem">🛗</div>
    <div>
        <h1>경남동부지사 톡톡AI</h1>
        <p>경남동부지사 업무지식 AI 질의응답 시스템</p>
    </div>
</div>
""", unsafe_allow_html=True)

if "messages"    not in st.session_state: st.session_state.messages    = []
if "vectorstore" not in st.session_state: st.session_state.vectorstore = None
if "doc_names"   not in st.session_state: st.session_state.doc_names   = []

GROQ_KEY      = st.secrets.get("GROQ_API_KEY", "")
FOLDER_ID     = st.secrets.get("FOLDER_ID", "")
PINECONE_KEY  = st.secrets.get("PINECONE_API_KEY", "")
PINECONE_IDX  = st.secrets.get("PINECONE_INDEX", "koelsa")

def get_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={"device": "cpu"}
    )

def get_drive_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    return build("drive", "v3", credentials=creds)

def get_drive_file_list():
    service = get_drive_service()
    results = service.files().list(
        q=f"'{FOLDER_ID}' in parents and mimeType='application/pdf' and trashed=false",
        fields="files(id, name, modifiedTime)"
    ).execute()
    return results.get("files", [])

def download_pdfs():
    from googleapiclient.http import MediaIoBaseDownload
    service = get_drive_service()
    files = get_drive_file_list()
    tmp_dir = Path(tempfile.mkdtemp())
    result = []
    for f in files:
        request = service.files().get_media(fileId=f["id"])
        path = tmp_dir / f["name"]
        with open(path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        result.append((str(path), f["name"]))
    return result

def extract_pdf(pdf_path, source_name):
    import pdfplumber
    from langchain_core.documents import Document
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables()
            table_lines = []
            for table in tables:
                for row in table:
                    if row:
                        cells = [str(c).strip() if c else "" for c in row]
                        line = " | ".join(cells)
                        if any(cells):
                            table_lines.append(line)
            if table_lines:
                text += "\n\n[표 내용]\n" + "\n".join(table_lines)
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"source": source_name, "page": i}
                ))
    return docs

def split_by_item(docs):
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    result = []
    item_pattern = re.compile(r"(?=\n\s*\d+\s*[‣▪▸►])")
    for doc in docs:
        text = doc.page_content
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", 0)
        year_match  = re.search(r"(20\d{2})년", text)
        order_match = re.search(r"제?(\d+)차", text)
        year  = year_match.group(1)  if year_match  else ""
        order = order_match.group(1) if order_match else ""
        parts = item_pattern.split(text)
        for part in parts:
            part = part.strip()
            if len(part) < 50:
                continue
            result.append(Document(
                page_content=part,
                metadata={"source": source, "page": page, "year": year, "order": order}
            ))
    if len(result) < len(docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000, chunk_overlap=500,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        result = splitter.split_documents(docs)
    return result

def get_pinecone_vectorstore(emb):
    from langchain_pinecone import PineconeVectorStore
    return PineconeVectorStore(index_name=PINECONE_IDX, embedding=emb, pinecone_api_key=PINECONE_KEY)

def is_pinecone_empty():
    from pinecone import Pinecone
    pc = Pinecone(api_key=PINECONE_KEY)
    idx = pc.Index(PINECONE_IDX)
    stats = idx.describe_index_stats()
    return stats.total_vector_count == 0

def build_and_save():
    from langchain_pinecone import PineconeVectorStore
    pdf_files = download_pdfs()
    if not pdf_files:
        return None, []
    all_docs, names = [], []
    for path, name in pdf_files:
        all_docs.extend(extract_pdf(path, name))
        names.append(name)
    chunks = split_by_item(all_docs)
    emb = get_embeddings()
    vs = PineconeVectorStore.from_documents(chunks, emb, index_name=PINECONE_IDX, pinecone_api_key=PINECONE_KEY)
    return vs, names

# 앱 시작 시 Pinecone 확인
if st.session_state.vectorstore is None:
    try:
        emb = get_embeddings()
        if is_pinecone_empty():
            with st.spinner("📚 첫 학습 중... (이후엔 빠르게 로드돼요)"):
                vs, names = build_and_save()
                st.session_state.vectorstore = vs
                st.session_state.doc_names   = names
        else:
            with st.spinner("💾 저장된 DB 불러오는 중..."):
                vs = get_pinecone_vectorstore(emb)
                files = get_drive_file_list()
                st.session_state.vectorstore = vs
                st.session_state.doc_names   = [f["name"] for f in files]
    except Exception as e:
        st.error(f"문서 로드 오류: {e}")

# 사이드바
with st.sidebar:
    st.markdown("### 📂 문서 현황")
    if st.session_state.vectorstore:
        st.markdown(f'<p class="status-ok">✅ {len(st.session_state.doc_names)}개 문서 학습 완료</p>', unsafe_allow_html=True)
        for n in st.session_state.doc_names:
            st.markdown(f"- 📄 `{n}`")
    else:
        st.markdown('<p class="status-warn">⚠️ 문서 로드 실패</p>', unsafe_allow_html=True)
    st.markdown("---")
    if st.button("🔄 문서 다시 학습", use_container_width=True):
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_KEY)
        pc.Index(PINECONE_IDX).delete(delete_all=True)
        st.session_state.vectorstore = None
        st.session_state.doc_names   = []
        st.rerun()
    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

col_chat, col_ex = st.columns([3, 1])

with col_ex:
    st.markdown("#### 💡 예시 질문")
    examples = [
        "승강장문 비상가이드 하부 설치 기준은?",
        "문이탈방지장치 인정 조건은?",
        "브레이크 교체하면 수시검사 대상이야?",
        "카간 칸막이 설치 기준은?",
        "정밀안전검사 상부 비상가이드 확인방법",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["preset_q"] = ex
            st.rerun()

with col_chat:
    chat_box = st.container(height=480)
    with chat_box:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center; color:#94a3b8; margin-top:80px;">
                <div style="font-size:3rem">🛗</div>
                <p style="margin-top:12px; font-size:0.95rem">
                    검사 기준에 대해 질문해보세요!
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    src_html = "".join(f'<span class="source-tag">📄 {s}</span>' for s in msg.get("sources", []))
                    st.markdown(
                        f'<div class="msg-bot">{msg["content"]}{"<br>" + src_html if src_html else ""}</div>',
                        unsafe_allow_html=True
                    )

    preset = st.session_state.pop("preset_q", "")
    user_input = st.chat_input("검사 기준에 대해 질문하세요...") or preset

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        if not st.session_state.vectorstore:
            ans, sources = "⚠️ 문서가 로드되지 않았어요.", []
        else:
            with st.spinner("검색 중..."):
                try:
                    from groq import Groq
                    client = Groq(api_key=GROQ_KEY)
                    rw = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": (
                                "당신은 승강기 검사 전문가입니다. "
                                "사용자 질문을 검사 문서에서 잘 검색되도록 다른 표현 4개로 바꿔주세요. "
                                "형식: 1. 검색어\n2. 검색어\n3. 검색어\n4. 검색어"
                            )},
                            {"role": "user", "content": user_input}
                        ],
                        temperature=0.3, max_tokens=200,
                    )
                    queries = [user_input]
                    for line in rw.choices[0].message.content.strip().split("\n"):
                        line = line.strip()
                        if line and line[0].isdigit():
                            q = line.split(".", 1)[-1].strip()
                            if q:
                                queries.append(q)
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                    seen, found = set(), []
                    for q in queries:
                        for d in retriever.invoke(q):
                            key = d.page_content[:100]
                            if key not in seen:
                                seen.add(key)
                                found.append(d)
                    found = found[:6]
                    context = "\n\n---\n\n".join(
                        f"[출처: {d.metadata.get('source','')} {d.metadata.get('page',0)+1}페이지 / {d.metadata.get('year','')}년 {d.metadata.get('order','')}차]\n{d.page_content}"
                        for d in found
                    )
                    sources = list({d.metadata.get("source", "") for d in found})
                    resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": (
                                "당신은 한국승강기안전공단 검사원을 지원하는 AI입니다. "
                                "제공된 문서 내용만 근거로 정확하게 답변하세요. "
                                "문서에 없으면 '해당 문서에서 확인되지 않습니다'라고 하세요. "
                                "수치(lx, mm, % 등)는 반드시 문서 그대로 인용하세요.\n\n"
                                "【중요 - 기준 우선순위】\n"
                                "동일 항목에 여러 연도 기준이 있을 경우:\n"
                                "1. 가장 최근 연도의 기준을 최우선 적용하세요.\n"
                                "2. 이전 기준은 종전 기준으로만 참고 언급하세요.\n"
                                "3. 답변 시 적용 기준의 연도/차수를 반드시 명시하세요.\n"
                                "4. 최신 기준이 이전을 변경했다면 그 사실을 알려주세요.\n\n"
                                f"[참고 문서]\n{context}"
                            )},
                            {"role": "user", "content": user_input}
                        ],
                        temperature=0.1, max_tokens=1024,
                    )
                    ans = resp.choices[0].message.content
                except Exception as e:
                    ans, sources = f"오류: {e}", []
        st.session_state.messages.append({"role": "assistant", "content": ans, "sources": sources})
        st.rerun()
