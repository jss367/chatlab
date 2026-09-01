import unittest

from model_runtime import validate_model_id


class ModelIdTests(unittest.TestCase):
    def test_accepts_hugging_face_model_id(self):
        self.assertEqual(
            validate_model_id(" allenai/Olmo-3-7B-Think "),
            "allenai/Olmo-3-7B-Think",
        )

    def test_rejects_local_and_incomplete_paths(self):
        for value in ("Olmo-3-7B-Think", "../model", "/tmp/model", "owner/model/extra"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_model_id(value)


if __name__ == "__main__":
    unittest.main()
