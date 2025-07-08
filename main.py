# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import router as kb_router         # 知识库接口

app = FastAPI()

# 前端跨域地址
origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080"
]

# 配置 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ 注册两个模块路由
app.include_router(kb_router, prefix="/api")

