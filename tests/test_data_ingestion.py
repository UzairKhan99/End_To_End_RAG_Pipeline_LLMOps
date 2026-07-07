import shutil
import unittest
from io import BytesIO
from pathlib import Path

from multi_doc_chat.src.document_ingestion.data_ingestion import ChatIngestor


class UploadedTextFile(BytesIO):
    name = "sample.txt"


class DataIngestionTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("tests/.tmp_ingestion")
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_uploaded_text_is_loaded_and_split(self):
        ingestor = ChatIngestor(
            temp_base=self.test_dir / "data",
            faiss_base=self.test_dir / "faiss",
            use_session_dirs=False,
        )
        uploaded_file = UploadedTextFile(
            b"Retrieval augmented generation uses relevant document context."
        )

        chunks = ingestor.build_retriever([uploaded_file])

        saved_file = self.test_dir / "data" / "sample.txt"
        print(f"\n[Ingestion] Saved file: {saved_file}")
        print(f"[Ingestion] Chunks created: {len(chunks)}")
        print(f"[Ingestion] First chunk: {chunks[0].page_content}")

        self.assertEqual(len(chunks), 1)
        self.assertIn("document context", chunks[0].page_content)
        self.assertTrue(saved_file.exists())


if __name__ == "__main__":
    unittest.main()
