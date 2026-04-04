from pydantic import BaseModel, EmailStr


class MemberAdd(BaseModel):
    email: EmailStr
    role: str  # admin, editor, viewer (not owner)


class MemberUpdate(BaseModel):
    role: str  # admin, editor, viewer (not owner)
