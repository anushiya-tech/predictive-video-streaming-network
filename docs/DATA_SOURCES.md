# Verified Phase 1 data sources

The executed pipeline downloaded the MIT-licensed Real-world-bandwidth-traces repository from https://github.com/confiwent/Real-world-bandwidth-traces. Its README identifies preprocessed public 3G/HSDPA and FCC throughput traces in timestamp/Mbps format. The source archive is retained under data/raw/ and is ignored by Git.

The implementation also supports original six-column Norway HSDPA and Belgium LTE mobile logs. The Norway format is documented at https://skuld.cs.umass.edu/traces/mmsys/2013/pathbandwidth/; Belgium logs and their format are documented at https://users.ugent.be/~jvdrhoof/dataset/. Their licence terms need confirmation before automated redistribution, so the downloader does not automate them. Place originals in data/raw/ for loader use.

