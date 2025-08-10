from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from typing import List, Dict

DATABASE_URL = "sqlite:///./themes.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def init_db_with_defaults() -> None:
    # 遅延インポートで循環参照を回避
    from models import Theme  # type: ignore

    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        defaults: List[Dict[str, str]] = [
            {
                "title": "休憩中の同僚との会話",
                "description": "オフィスの休憩スペースで同僚と交わすカジュアルな会話。軽い雑談、仕事の合間の話題、リフレッシュのやり取り。",
                "system_prompt": (
                    "You are a friendly coworker chatting during a short break at the office. "
                    "Keep it casual and supportive. Ask simple follow-up questions, avoid long monologues, "
                    "and keep responses within 1-2 short sentences. Stay on the breakroom theme. Respond in English only."
                ),
            },
            {
                "title": "朝の会話",
                "description": "朝の挨拶、体調や予定の確認、1日の始まりに交わすシンプルで前向きな会話。",
                "system_prompt": (
                    "You are a warm, encouraging partner having a short morning conversation. "
                    "Greet politely, keep a positive tone, ask brief follow-ups, "
                    "and keep responses within 1-2 short sentences. Stay on the morning routine theme. Respond in English only."
                ),
            },
        ]

        for t in defaults:
            existing = session.query(Theme).filter_by(title=t["title"]).first()
            if existing is None:
                session.add(Theme(**t))
        session.commit()
    finally:
        session.close()


