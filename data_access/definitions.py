import sys
from typing import Iterable
from zipfile import ZipFile
from saf import Sentence, Token

DSR_FILES = {
    "wikipedia": "data/DSR/Wikipedia/WKP_DSR_model_CSV.zip",
    "wiktionary": "data/DSR/Wikitionary/WKT_DSR_model_CSV.zip",
    "wordnet": "data/DSR/WordNet/WN_DSR_model_CSV.zip",
}


class DefinitionSemanticRoleCorpus(Iterable[Sentence]):
    def __init__(self, path: str):
        if (path in DSR_FILES):
            path = DSR_FILES[path]

        dsr_zip = ZipFile(path)
        self._source = dsr_zip.open(dsr_zip.namelist()[0])

        self._size = 0

        for line in self._source:
            self._size += 1

        self._source.seek(0)

    def __iter__(self):
        return DefinitionSemanticRoleCorpusIterator(self)

    def __len__(self):
        return self._size


class DefinitionSemanticRoleCorpusIterator:
    def __init__(self, dsrc: DefinitionSemanticRoleCorpus):
        self._dsrc = dsrc
        self._sent_gen = self.sent_generator()

    def __next__(self):
        return next(self._sent_gen)

    def sent_generator(self):
        k = 0
        sentence_buffer = [None]
        while (sentence_buffer):
            sentence_buffer = list()
            while (not sentence_buffer):
                try:
                    line_bytes = next(self._dsrc._source)
                    line_bytes = line_bytes.replace(b"\r", b"\\r").replace(b"\t", b"\\t")
                    line_bytes = line_bytes.replace(b"\N", b"\\N").replace(b"\c", b"\\c")
                    line_bytes = line_bytes.replace(b"\i", b"\\i")
                    line = line_bytes.decode("unicode_escape")
                    line = line.strip().replace("&amp;", "&").replace("&quot;", "\"")
                    fields = line.split(";")

                    terms = fields[2].split(", ")

                    for term in terms:
                        sentence = Sentence()
                        sentence.annotations["id"] = fields[0]
                        sentence.annotations["POS"] = fields[1]
                        sentence.annotations["definiendum"] = term
                        sentence.annotations["definition"] = fields[3]

                        for i in range(4, len(fields)):
                            segment_role = fields[i].split("/")
                            segment = "/".join(segment_role[:-1])
                            role = segment_role[-1]
                            for tok in segment.split():
                                token = Token()
                                token.surface = tok
                                token.annotations["DSR"] = role
                                sentence.tokens.append(token)


                        sentence_buffer.append(sentence)

                except UnicodeDecodeError:
                    print("Decode error", file=sys.stderr)
                except StopIteration:
                    break

            for sentence in sentence_buffer:
                # print([t.surface for t in sentence.tokens])

                yield sentence


if __name__ == "__main__":
    dsrc = DefinitionSemanticRoleCorpus("wordnet")
    print("Corpus size:", len(dsrc))

    i = 0
    for sent in dsrc:
        print("Sent annotations:", sent.annotations)
        print("Token annotations:", [(token.surface, token.annotations["DSR"]) for token in sent.tokens])
        i += 1
        if (i > 10):
           break
