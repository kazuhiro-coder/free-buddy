from sqlalchemy import Column, Integer, String, Text
from db import Base


class Theme(Base):
    __tablename__ = "themes"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String(100), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=False)
    system_prompt = Column(Text, nullable=False)


