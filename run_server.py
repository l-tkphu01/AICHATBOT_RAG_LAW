import uvicorn
from dotenv import load_dotenv

if __name__ == "__main__":
    # Đảm bảo các biến môi trường được tải
    load_dotenv()
    print("="*60)
    print("🚀 KHỞI ĐỘNG FASTAPI SERVER CHO LEGAL RAG 🚀")
    print("="*60)
    
    # Khởi chạy ứng dụng với Uvicorn qua module app.api.main
    uvicorn.run(
        "app.api.main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True  # Bật tính năng auto-reload khi code thay đổi (chỉ dùng cho môi trường dev)
    )
