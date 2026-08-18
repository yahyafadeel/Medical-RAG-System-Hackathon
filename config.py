from pathlib import Path

# تحديد مسار فولدر الداتا اللي جواه الملف الطبي
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# إعدادات الـ Chunking (أرقام مبدئية هنعدلها ونجربها بعدين)
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50