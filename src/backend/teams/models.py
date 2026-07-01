from pydantic import BaseModel, EmailStr


class TeamCreate(BaseModel):
    name: str
    description: str | None = None


class TeamUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class InviteCreate(BaseModel):
    email: EmailStr
    role: str = "member"  # 'admin' | 'member'


class MemberRoleUpdate(BaseModel):
    role: str  # 'admin' | 'member'
