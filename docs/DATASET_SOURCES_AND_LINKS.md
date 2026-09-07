# Dataset Sources, Web Portals, and Download Links

This document provides a comprehensive, centralized catalog of every online source, repository, dataset download portal, and API endpoint used to extract both the **NLP (Domain 1)** and **COVID-19 CS-Adjacent (Domain 2)** scientific literature corpora.

---

## 1. Domain 1: Natural Language Processing (NLP) Corpus (2018–2024)

### A. ACL Anthology (Primary Peer-Reviewed NLP Venue Source)
The ACL Anthology hosts accepted papers and proceedings for the premier computational linguistics and NLP conferences.

* **Official Web Portal:** [https://aclanthology.org](https://aclanthology.org)
* **Official GitHub Data Repository (Source XML):** [https://github.com/acl-org/acl-anthology](https://github.com/acl-org/acl-anthology)
* **Raw XML Data Directory (Master Branch):** [https://github.com/acl-org/acl-anthology/tree/master/data/xml](https://github.com/acl-org/acl-anthology/tree/master/data/xml)
* **Raw XML Direct URL Pattern:** `https://raw.githubusercontent.com/acl-org/acl-anthology/master/data/xml/{collection_id}.xml`
* **Anthology GitHub Commits API (Reproducibility SHA):** [https://api.github.com/repos/acl-org/acl-anthology/commits/master](https://api.github.com/repos/acl-org/acl-anthology/commits/master)

#### Individual Conference Venues within ACL Anthology:
* **ACL (Annual Meeting of the Association for Computational Linguistics):** [https://aclanthology.org/venues/acl/](https://aclanthology.org/venues/acl/)
* **EMNLP (Empirical Methods in Natural Language Processing):** [https://aclanthology.org/venues/emnlp/](https://aclanthology.org/venues/emnlp/)
* **NAACL (North American Chapter of the ACL):** [https://aclanthology.org/venues/naacl/](https://aclanthology.org/venues/naacl/)
* **Findings of the ACL (Catch-all for ACL / EMNLP / NAACL):** [https://aclanthology.org/venues/findings/](https://aclanthology.org/venues/findings/)
* **COLING (International Conference on Computational Linguistics):** [https://aclanthology.org/venues/coling/](https://aclanthology.org/venues/coling/)
* **CoNLL (Conference on Computational Natural Language Learning):** [https://aclanthology.org/venues/conll/](https://aclanthology.org/venues/conll/)
* **EACL (European Chapter of the ACL):** [https://aclanthology.org/venues/eacl/](https://aclanthology.org/venues/eacl/)
* **TACL (Transactions of the Association for Computational Linguistics):** [https://aclanthology.org/venues/tacl/](https://aclanthology.org/venues/tacl/)

---

### B. arXiv (Computation and Language — `cs.CL`)
Used for early-year preprint coverage and cutting-edge NLP research prior to formal venue publication.

* **arXiv Computation and Language (`cs.CL`) Recent Archive:** [https://arxiv.org/list/cs.CL/recent](https://arxiv.org/list/cs.CL/recent)
* **arXiv Computer Science Main Archive:** [https://arxiv.org/archive/cs](https://arxiv.org/archive/cs)
* **arXiv Advanced Search Engine:** [https://arxiv.org/search/advanced](https://arxiv.org/search/advanced)
* **arXiv REST / OAI-PMH API Endpoint:** `http://export.arxiv.org/api/query`
* **arXiv API User Manual & Specs:** [https://arxiv.org/help/api/user-manual](https://arxiv.org/help/api/user-manual)

---

### C. Semantic Scholar (Academic Citation Graph & Metadata Enrichment)
Used to retrieve citation counts, influential citations, normalized venue names, and cross-reference identifiers.

* **Semantic Scholar Main Portal:** [https://www.semanticscholar.org](https://www.semanticscholar.org)
* **Semantic Scholar Academic API Developer Hub:** [https://www.semanticscholar.org/product/api](https://www.semanticscholar.org/product/api)
* **Graph API Official Documentation:** [https://api.semanticscholar.org/api-docs/graph](https://api.semanticscholar.org/api-docs/graph)
* **Paper Search API Endpoint:** `https://api.semanticscholar.org/graph/v1/paper/search`
* **Paper Batch Lookup API Endpoint:** `https://api.semanticscholar.org/graph/v1/paper/batch`

---

## 2. Domain 2: COVID-19 CS-Adjacent Corpus (2019–2024)

### A. CORD-19 (COVID-19 Open Research Dataset)
Curated by the Allen Institute for AI (AI2) in partnership with the White House OSTP, NIH, Chan Zuckerberg Initiative, and Georgetown CSET.

* **Kaggle CORD-19 Research Challenge Dataset:** [https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge)
* **Official AI2 CORD-19 Project Portal:** [https://www.semanticscholar.org/cord19](https://www.semanticscholar.org/cord19)
* **Historical Releases & Direct CSV / JSON Download Index (AWS S3):** [https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases.html](https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases.html)
* **CORD-19 Canonical Paper Overview:** [https://www.semanticscholar.org/paper/CORD-19%3A-The-Covid-19-Open-Research-Dataset-Wang-Lo/0150965d1d6439ab88cfa861ef1c0e359050d5e8](https://www.semanticscholar.org/paper/CORD-19%3A-The-Covid-19-Open-Research-Dataset-Wang-Lo/0150965d1d6439ab88cfa861ef1c0e359050d5e8)

---

### B. Europe PMC (Biomedical & Life Sciences Literature)
Comprehensive database providing access to worldwide biomedical research literature, abstracts, and full texts.

* **Europe PMC Main Search Portal:** [https://europepmc.org](https://europepmc.org)
* **Europe PMC REST Search API Endpoint:** `https://www.ebi.ac.uk/europepmc/webservices/rest/search`
* **Europe PMC Web Services Documentation:** [https://europepmc.org/RestfulWebService](https://europepmc.org/RestfulWebService)

---

### C. PubMed / NCBI (National Center for Biotechnology Information)
The premier index of medical and biomedical sciences papers, maintained by the United States National Library of Medicine (NLM).

* **PubMed Web Portal:** [https://pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov)
* **NCBI Entrez E-Utilities Search (`esearch`):** `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi`
* **NCBI Entrez E-Utilities Summary (`esummary`):** `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi`
* **NCBI E-Utilities Developer Guide:** [https://www.ncbi.nlm.nih.gov/books/NBK25501/](https://www.ncbi.nlm.nih.gov/books/NBK25501/)

---

### D. OpenAlex (Global Open Scholarly Knowledge Graph)
A fully open, comprehensive catalog of global academic publications, authors, institutions, and citation networks.

* **OpenAlex Web Portal:** [https://openalex.org](https://openalex.org)
* **OpenAlex Works REST API Endpoint:** `https://api.openalex.org/works`
* **OpenAlex Official Developer Documentation:** [https://docs.openalex.org](https://docs.openalex.org)

---

## 3. Subtopic Web Searches (COVID-19 CS-Adjacent Focus Areas)

Direct query links for exploring papers across the 7 computer-science-adjacent subtopics:

1. **Misinformation & Fake News Detection:**
   * PubMed: [https://pubmed.ncbi.nlm.nih.gov/?term=COVID-19+misinformation+detection](https://pubmed.ncbi.nlm.nih.gov/?term=COVID-19+misinformation+detection)
   * Europe PMC: [https://europepmc.org/search?query=COVID-19%20misinformation%20fake%20news](https://europepmc.org/search?query=COVID-19%20misinformation%20fake%20news)
   * arXiv: [https://arxiv.org/search/?query=COVID-19+misinformation&searchtype=all](https://arxiv.org/search/?query=COVID-19+misinformation&searchtype=all)

2. **Contact Tracing & Exposure Notification:**
   * PubMed: [https://pubmed.ncbi.nlm.nih.gov/?term=COVID-19+contact+tracing+exposure+notification](https://pubmed.ncbi.nlm.nih.gov/?term=COVID-19+contact+tracing+exposure+notification)
   * Europe PMC: [https://europepmc.org/search?query=COVID-19%20contact%20tracing](https://europepmc.org/search?query=COVID-19%20contact%20tracing)

3. **Conversational Agents & Symptom-Checker Chatbots:**
   * PubMed: [https://pubmed.ncbi.nlm.nih.gov/?term=COVID-19+chatbot+conversational+agent](https://pubmed.ncbi.nlm.nih.gov/?term=COVID-19+chatbot+conversational+agent)
   * Europe PMC: [https://europepmc.org/search?query=COVID-19%20chatbot%20conversational%20agent](https://europepmc.org/search?query=COVID-19%20chatbot%20conversational%20agent)

4. **Epidemiological Modeling & Machine Learning (SEIR / Forecasting):**
   * Europe PMC: [https://europepmc.org/search?query=COVID-19%20epidemiological%20forecasting%20model%20machine%20learning](https://europepmc.org/search?query=COVID-19%20epidemiological%20forecasting%20model%20machine%20learning)
   * arXiv: [https://arxiv.org/search/?query=COVID-19+SEIR+forecasting&searchtype=all](https://arxiv.org/search/?query=COVID-19+SEIR+forecasting&searchtype=all)

5. **Health Informatics & Electronic Health Records (EHR):**
   * PubMed: [https://pubmed.ncbi.nlm.nih.gov/?term=COVID-19+electronic+health+records+informatics](https://pubmed.ncbi.nlm.nih.gov/?term=COVID-19+electronic+health+records+informatics)
   * Europe PMC: [https://europepmc.org/search?query=COVID-19%20electronic%20health%20records%20EHR%20informatics](https://europepmc.org/search?query=COVID-19%20electronic%20health%20records%20EHR%20informatics)

6. **NLP & Literature Text Mining:**
   * Europe PMC: [https://europepmc.org/search?query=COVID-19%20natural%20language%20processing%20text%20mining](https://europepmc.org/search?query=COVID-19%20natural%20language%20processing%20text%20mining)
   * Semantic Scholar: [https://www.semanticscholar.org/search?q=COVID-19%20NLP%20text%20mining](https://www.semanticscholar.org/search?q=COVID-19%20NLP%20text%20mining)

7. **Social Media Sentiment Analysis & Public Surveillance:**
   * Europe PMC: [https://europepmc.org/search?query=COVID-19%20social%20media%20sentiment%20analysis%20twitter](https://europepmc.org/search?query=COVID-19%20social%20media%20sentiment%20analysis%20twitter)
   * PubMed: [https://pubmed.ncbi.nlm.nih.gov/?term=COVID-19+social+media+sentiment+analysis](https://pubmed.ncbi.nlm.nih.gov/?term=COVID-19+social+media+sentiment+analysis)

---

## 4. Pipeline Script to Dataset Source Mapping

| Script | Primary Data Source | Web / Endpoint URL |
| :--- | :--- | :--- |
| `01_collect_acl_anthology.py` | ACL Anthology XML on GitHub | [https://github.com/acl-org/acl-anthology](https://github.com/acl-org/acl-anthology) |
| `02_collect_semantic_scholar.py` | Semantic Scholar Academic API | [https://www.semanticscholar.org/product/api](https://www.semanticscholar.org/product/api) |
| `03_collect_arxiv.py` | arXiv API (`cs.CL` category) | [https://arxiv.org/list/cs.CL/recent](https://arxiv.org/list/cs.CL/recent) |
| `04_filter_cord19.py` | CORD-19 Archive & S2 API | [https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge](https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge) |
| `05_collect_covid_papers.py` | Europe PMC, PubMed, OpenAlex, arXiv | [https://europepmc.org](https://europepmc.org), [https://pubmed.ncbi.nlm.nih.gov](https://pubmed.ncbi.nlm.nih.gov), [https://openalex.org](https://openalex.org) |
| `06_preprocess_and_sample.py` | Preprocessing, Filtering, Scoring | Stratifies & deduplicates raw pools into final sampled corpora |
