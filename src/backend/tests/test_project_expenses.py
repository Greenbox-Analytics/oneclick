"""Tests for project expense tracking endpoints (net-vs-gross royalty support).

Acceptance criteria:
1. List expenses (members can read; non-members 403) with linked work_ids attached.
2. Create expense (editors+ only; viewers 403; validation on amount).
3. Update expense (editors+ only).
4. Delete expense (editors+ only).
"""

from unittest.mock import MagicMock

from tests.conftest import TEST_USER_ID, MockQueryBuilder

PROJECT_ID = "proj-00000000-0000-0000-0000-000000000001"
EXPENSE_ID = "exp-00000000-0000-0000-0000-000000000001"
WORK_ID = "work-00000000-0000-0000-0000-000000000001"

EXPENSE_RECORD = {
    "id": EXPENSE_ID,
    "project_id": PROJECT_ID,
    "created_by": TEST_USER_ID,
    "description": "Studio time",
    "amount": 500.0,
    "category": "studio",
    "incurred_on": "2026-06-01",
    "created_at": "2026-06-01T00:00:00+00:00",
}


def _builder(data):
    b = MockQueryBuilder()
    count = len(data) if isinstance(data, list) else (1 if data else 0)
    b.execute.return_value = MagicMock(data=data, count=count)
    return b


def _seq_side_effect(sequences: list):
    idx = [0]

    def _side_effect(name):
        data = sequences[idx[0]] if idx[0] < len(sequences) else []
        idx[0] += 1
        return _builder(data)

    return _side_effect


# ===========================================================================
# GET /projects/{project_id}/expenses
# ===========================================================================


class TestListExpenses:
    def test_returns_expenses_with_work_ids(self, client, mock_supabase):
        """Returns expenses, each enriched with its linked work_ids."""
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                {"role": "viewer"},  # get_user_role
                [EXPENSE_RECORD],  # project_expenses listing
                [{"expense_id": EXPENSE_ID, "work_id": WORK_ID}],  # join links
            ]
        )

        response = client.get(f"/projects/{PROJECT_ID}/expenses")

        assert response.status_code == 200
        expenses = response.json()["expenses"]
        assert len(expenses) == 1
        assert expenses[0]["id"] == EXPENSE_ID
        assert expenses[0]["work_ids"] == [WORK_ID]

    def test_returns_empty_work_ids_when_untagged(self, client, mock_supabase):
        """An untagged expense gets work_ids=[]."""
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                {"role": "owner"},
                [EXPENSE_RECORD],
                [],  # no links
            ]
        )

        response = client.get(f"/projects/{PROJECT_ID}/expenses")

        assert response.status_code == 200
        assert response.json()["expenses"][0]["work_ids"] == []

    def test_returns_403_when_not_a_member(self, client, mock_supabase):
        mock_supabase.table.side_effect = _seq_side_effect([None])  # get_user_role → None

        response = client.get(f"/projects/{PROJECT_ID}/expenses")

        assert response.status_code == 403


# ===========================================================================
# POST /projects/{project_id}/expenses
# ===========================================================================


class TestCreateExpense:
    def test_editor_creates_expense_with_work_links(self, client, mock_supabase):
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                {"role": "editor"},  # get_user_role
                [EXPENSE_RECORD],  # project_expenses INSERT
                [],  # project_expense_works DELETE (replace)
                [],  # project_expense_works INSERT
            ]
        )

        response = client.post(
            f"/projects/{PROJECT_ID}/expenses",
            json={
                "description": "Studio time",
                "amount": 500.0,
                "category": "studio",
                "incurred_on": "2026-06-01",
                "work_ids": [WORK_ID],
            },
        )

        assert response.status_code == 200
        expense = response.json()["expense"]
        assert expense["id"] == EXPENSE_ID
        assert expense["work_ids"] == [WORK_ID]

    def test_returns_403_when_viewer(self, client, mock_supabase):
        mock_supabase.table.side_effect = _seq_side_effect([{"role": "viewer"}])

        response = client.post(
            f"/projects/{PROJECT_ID}/expenses",
            json={"description": "x", "amount": 10.0},
        )

        assert response.status_code == 403

    def test_returns_422_for_negative_amount(self, client, mock_supabase):
        response = client.post(
            f"/projects/{PROJECT_ID}/expenses",
            json={"description": "x", "amount": -5.0},
        )

        assert response.status_code == 422

    def test_returns_422_when_amount_missing(self, client, mock_supabase):
        response = client.post(
            f"/projects/{PROJECT_ID}/expenses",
            json={"description": "x"},
        )

        assert response.status_code == 422


# ===========================================================================
# PUT /projects/{project_id}/expenses/{expense_id}
# ===========================================================================


class TestUpdateExpense:
    def test_editor_updates_expense(self, client, mock_supabase):
        updated = {**EXPENSE_RECORD, "amount": 750.0}
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                {"role": "owner"},  # get_user_role
                [updated],  # project_expenses UPDATE (not read)
                [],  # project_expense_works DELETE (work_ids=[] replace)
                updated,  # project_expenses select maybe_single
                [],  # _attach_expense_works join select
            ]
        )

        response = client.put(
            f"/projects/{PROJECT_ID}/expenses/{EXPENSE_ID}",
            json={"amount": 750.0, "work_ids": []},
        )

        assert response.status_code == 200
        assert response.json()["expense"]["amount"] == 750.0

    def test_returns_403_when_viewer(self, client, mock_supabase):
        mock_supabase.table.side_effect = _seq_side_effect([{"role": "viewer"}])

        response = client.put(
            f"/projects/{PROJECT_ID}/expenses/{EXPENSE_ID}",
            json={"amount": 10.0},
        )

        assert response.status_code == 403


# ===========================================================================
# DELETE /projects/{project_id}/expenses/{expense_id}
# ===========================================================================


class TestDeleteExpense:
    def test_editor_deletes_expense(self, client, mock_supabase):
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                {"role": "editor"},  # get_user_role
                [],  # DELETE
            ]
        )

        response = client.delete(f"/projects/{PROJECT_ID}/expenses/{EXPENSE_ID}")

        assert response.status_code == 200
        assert response.json()["deleted"] == EXPENSE_ID

    def test_returns_403_when_viewer(self, client, mock_supabase):
        mock_supabase.table.side_effect = _seq_side_effect([{"role": "viewer"}])

        response = client.delete(f"/projects/{PROJECT_ID}/expenses/{EXPENSE_ID}")

        assert response.status_code == 403


# ===========================================================================
# GET /expenses/summary (cross-project rollup)
# ===========================================================================

ARTIST_ID = "art-00000000-0000-0000-0000-000000000001"


class TestExpensesSummary:
    def test_returns_enriched_rows_across_projects(self, client, mock_supabase):
        """Summary joins project + artist names and flags tagged expenses."""
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                [{"project_id": PROJECT_ID}],  # project_members (memberships)
                [{"id": PROJECT_ID, "name": "Midnight EP", "artist_id": ARTIST_ID}],  # projects
                [{"id": ARTIST_ID, "name": "Nova"}],  # artists
                [EXPENSE_RECORD],  # project_expenses
                [{"expense_id": EXPENSE_ID}],  # project_expense_works (tagged)
            ]
        )

        response = client.get("/expenses/summary")

        assert response.status_code == 200
        rows = response.json()["expenses"]
        assert len(rows) == 1
        row = rows[0]
        assert row["project_id"] == PROJECT_ID
        assert row["project_name"] == "Midnight EP"
        assert row["artist_name"] == "Nova"
        assert row["amount"] == 500.0
        assert row["is_tagged"] is True

    def test_untagged_expense_flagged_false(self, client, mock_supabase):
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                [{"project_id": PROJECT_ID}],
                [{"id": PROJECT_ID, "name": "Midnight EP", "artist_id": ARTIST_ID}],
                [{"id": ARTIST_ID, "name": "Nova"}],
                [EXPENSE_RECORD],
                [],  # no work links
            ]
        )

        response = client.get("/expenses/summary")

        assert response.status_code == 200
        assert response.json()["expenses"][0]["is_tagged"] is False

    def test_empty_when_no_memberships(self, client, mock_supabase):
        mock_supabase.table.side_effect = _seq_side_effect([[]])  # no memberships

        response = client.get("/expenses/summary")

        assert response.status_code == 200
        assert response.json()["expenses"] == []

    def test_empty_when_no_expenses(self, client, mock_supabase):
        mock_supabase.table.side_effect = _seq_side_effect(
            [
                [{"project_id": PROJECT_ID}],
                [{"id": PROJECT_ID, "name": "Midnight EP", "artist_id": ARTIST_ID}],
                [{"id": ARTIST_ID, "name": "Nova"}],
                [],  # no expenses
            ]
        )

        response = client.get("/expenses/summary")

        assert response.status_code == 200
        assert response.json()["expenses"] == []
