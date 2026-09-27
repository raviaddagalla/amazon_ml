# EDA Findings — Business Entity Resolution

## 1. Dataset Scale

| Dataset | Total Rows | US | India | France |
|---------|------------|-----|-------|--------|
| train_s1 | 2,206,821 | 1,323,633 | 883,188 | — |
| train_s2 | 5,034,616 | 3,016,817 | 2,017,799 | — |
| train_s3 | 5,285,603 | 3,170,056 | 2,115,547 | — |
| test_s1 | 1,732,544 | 663,106 | 809,986 | 259,452 |
| test_s2 | 4,887,273 | 1,871,330 | 2,312,565 | 703,378 |
| test_s3 | 5,082,316 | 1,945,701 | 2,405,000 | 731,615 |

**Key observation**: Each S1 entity has a candidate pool of ~5-6M records per source (S2+S3 combined ~10M per country split). The old training sampled only 80K per source — a **~65x gap** that explains the inflated training validation score.

## 2. Ground Truth Match Distribution

| Matches per S1 entity | Count |
|----------------------|-------|
| 0 (singleton) | 123,247 (5.6%) |
| 1 | 119,157 (5.4%) |
| 2 | 375,212 (17.0%) |
| 3 | 530,841 (24.1%) |
| 4 | 484,115 (21.9%) |
| 5 | 321,957 (14.6%) |
| 6 | 164,868 (7.5%) |
| 7 | 63,968 (2.9%) |
| 8 | 18,680 (0.8%) |
| 9+ | 4,776 (0.2%) |

- Most entities have 2-5 matches (median ~3)
- 5.6% are true singletons — must predict empty for those or pay full F0.5 penalty
- Up to 11 matches per entity (rare)

## 3. Column Schema

All source files: `entity_id, business_name, business_address, country`
Ground truth: `source1_entity_id, matched_entity_ids` (comma-separated)

## 4. True-Positive Pair Patterns (from 200 sample pairs)

### 4.1 Case variations
Extremely common. S1 typically has proper case; S2/S3 may be ALL CAPS, all lowercase, or mixed.
- `"Galore Aluminium (India) Private Limited"` ↔ `"galore aluminium (india) private limited"`

### 4.2 Abbreviation mismatches
Very common for both names and addresses:
- `"Street"` ↔ `"St"`, `"Road"` ↔ `"Rd"`, `"Avenue"` ↔ `"Ave"`
- `"Private Limited"` ↔ `"Pvt Ltd"`
- `"Lucky Barbershop"` (S1) vs `"Lucky Barbershop Ltd"` (S2)

### 4.3 Special characters / Unicode
S3 in particular introduces diacritics, hyphens, and ID suffixes:
- `"Lucky Barbershop"` → `"Lücky-Barbershop"` (S3 adds diacritics/hyphens)
- S3 appends metadata like `"(ID: 35808)"` after business name

### 4.4 Address format variations
- Full state name vs abbreviation: `"West Virginia"` ↔ `"WV"`
- Locality suffixes: `"CHARLES TOWN"` ↔ `"Charles TOWN CDP"`
- Plot/building prefixes: `"208-94A, Ramamandira Road"` ↔ `"A-208-94A, RAMAMANDIRA ROAD"`
- India: `"Mysore, Karnataka"` ↔ `"Mysore, KA"`

### 4.5 Word-order transpositions
Occasionally seen, especially in Indian business names.

### 4.6 DBA / trade name differences
`"Future Management Pvt Ltd"` can appear as `"Future Management"` or `"Future Mgmt Private Limited"` across sources.

## 5. France Domain-Shift Risk

> [!WARNING]
> France appears ONLY in test, never in training. This is a cold-start scenario.

- Test France S1: 259,452 entities
- Test France S2: 703,378 candidates
- Test France S3: 731,615 candidates

**Critical implications:**
- French corporate suffixes (`SARL`, `SAS`, `SASU`, `EURL`, `SCI`) must be in stopwords
- French street types (`rue`, `avenue`, `boulevard`, `allée`, `impasse`, `chemin`, `place`, `passage`, `voie`) and directionals must be handled
- `CEDEX` (postal routing code) and French postal codes (5 digits starting with department codes) need support
- `de`, `du`, `des`, `la`, `le`, `les` are common French prepositions in addresses/names — should be stopwords
- The model must be language-agnostic enough to handle French without having seen training examples

## 6. Implications for Pipeline Upgrades

1. **Abbreviation dictionary** (Phase 2): Must cover US street abbreviations, Indian state abbreviations, French address terms, and corporate suffixes for all three countries.
2. **Stopwords**: Extended list needed (already partially done in current config.py).
3. **ID suffix stripping**: S3 appends `(ID: XXXXX)` — should be stripped in preprocessing.
4. **Sorted-token signatures**: Will help with word-order transposition cases.
5. **Address component extraction**: Street numbers and postal codes are strong matching signals — need robust, country-agnostic extraction.
6. **Missing address handling**: Current `a_set=a_part=50.0` default is arbitrary — must be replaced with proper indicator features.
