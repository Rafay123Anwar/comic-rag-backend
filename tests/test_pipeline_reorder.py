import unittest
from unittest import mock
from pathlib import Path

from app.routes.comic import run_background_comic_analysis


class TestPipelineReordering(unittest.TestCase):
    def test_pipeline_execution_order(self):
        call_order = []

        fake_pages = [
            {
                "page_number": 1,
                "filename": "page_001.jpg",
                "image_path": "storage/comics/test-id/pages/page_001.jpg",
                "thumbnail_path": "storage/comics/test-id/thumbnails/thumb_p001.jpg"
            }
        ]

        def fake_upload_assets(*args, **kwargs):
            call_order.append("upload_comic_assets_immediately")
            return [
                {
                    "page_number": 1,
                    "filename": "page_001.jpg",
                    "image_path": "storage/comics/test-id/pages/page_001.jpg",
                    "thumbnail_path": "storage/comics/test-id/thumbnails/thumb_p001.jpg",
                    "image_url": "https://supabase.co/signed/page_001.jpg",
                    "thumbnail_url": "https://supabase.co/signed/thumb_p001.jpg",
                    "image_storage_path": "user/u1/comics/test-id/pages/page_001.jpg",
                    "thumbnail_storage_path": "user/u1/comics/test-id/thumbnails/thumb_p001.jpg"
                }
            ]

        def fake_save_json(*args, **kwargs):
            status = kwargs.get("status")
            call_order.append(f"save_comic_json_{status}")
            return {"comic": {"id": "test-id", "status": status}}

        def fake_analyze_pages(*args, **kwargs):
            call_order.append("analyze_pages")
            on_complete = kwargs.get("on_page_complete")
            res = {
                "page_number": 1,
                "status": "success",
                "analysis": {"text": {"full_text": "Hello Comic"}}
            }
            if on_complete:
                on_complete(res, 1, 1)
            return [res]

        def fake_ingest_page(*args, **kwargs):
            call_order.append("ingest_page_to_rag")

        def fake_ingest_comic(*args, **kwargs):
            call_order.append("ingest_comic_to_rag")

        # Mock SessionLocal and DB
        mock_db = mock.MagicMock()
        mock_comic_row = mock.MagicMock()
        mock_db.query.return_value.filter.return_value.all.return_value = []
        mock_db.query.return_value.filter.return_value.first.return_value = mock_comic_row

        with mock.patch("app.routes.comic.upload_comic_assets_immediately", side_effect=fake_upload_assets), \
             mock.patch("app.routes.comic.save_comic_json", side_effect=fake_save_json), \
             mock.patch("app.routes.comic.analyze_pages", side_effect=fake_analyze_pages), \
             mock.patch("app.routes.comic.ingest_page_to_rag", side_effect=fake_ingest_page), \
             mock.patch("app.routes.comic.ingest_comic_to_rag", side_effect=fake_ingest_comic), \
             mock.patch("app.routes.comic.SessionLocal", return_value=mock_db):

            run_background_comic_analysis(
                comic_id="test-id",
                pages=fake_pages,
                comic_name="Test Comic",
                source_format="cbr",
                initial_pages=fake_pages,
                user_id="u1"
            )

        # Verify exact sequence
        # 1. Immediate upload to Supabase Storage
        self.assertEqual(call_order[0], "upload_comic_assets_immediately")
        
        # 2. Save initial records to DB with status='processing'
        self.assertEqual(call_order[1], "save_comic_json_processing")
        
        # 3. Analyze pages with EdenAI
        self.assertIn("analyze_pages", call_order)
        idx_analyze = call_order.index("analyze_pages")
        
        # Asset upload and initial save MUST happen BEFORE analyze_pages
        self.assertLess(call_order.index("upload_comic_assets_immediately"), idx_analyze)
        self.assertLess(call_order.index("save_comic_json_processing"), idx_analyze)
        
        # 4. RAG ingestion occurs
        self.assertIn("ingest_page_to_rag", call_order)
        
        # 5. Final completion save
        self.assertEqual(call_order[-1], "save_comic_json_completed")
        self.assertEqual(mock_comic_row.status, "completed")


if __name__ == "__main__":
    unittest.main()
