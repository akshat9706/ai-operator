> **Sample output from the OFFLINE demo** (`python -m operator_agent demo --scenario mumbai-flaky`). A scripted test double played the model; this shows the platform's evidence report, not model behaviour. Ports and timing vary per run.

# Operator run report

**Request:** Submit my expenses for the Mumbai trip.

**Requested by:** Priya Sharma (priya.sharma)

**Outcome:** COMPLETED and independently verified

*48 model turns, 47 tool calls, 0.5s, generated 2026-10-13 10:12 UTC*

## Summary from the operator

Expense report #1 'Trip - Mumbai - Sep 2026' submitted for priya.sharma: 6 lines, total 26030 INR, cost center CC-2207.

Claimed:
- 2026-09-14 Transport IndiGo Airlines: 6420 (01_indigo_flight)
- 2026-09-14 Transport Meru Cabs: 1150 (02_meru_taxi_airport)
- 2026-09-14 Lodging Taj Lands End, Mumbai: 15000 (03_taj_hotel_folio)
- 2026-09-14 Meals Trishna Restaurant: 1500 (04_trishna_dinner)
- 2026-09-15 Meals Kyani & Co.: 980 (05_kyani_lunch)
- 2026-09-16 Transport Uber: 980 (06_uber_return)

Left out:
- 03_taj_hotel_folio: Minibar (850) not reimbursable
- 04_trishna_dinner: Kingfisher beer x2 (alcohol) (890) not reimbursable
- 04b_trishna_dinner_rescan: duplicate of an already-claimed bill (TR-20993)
- 07_fabindia_gifts: personal purchase/gift (not reimbursable)
- 04_trishna_dinner: 250 over the daily meal cap of 1500

## Independent verification

Result: **PASSED**. Checks ran from a fresh session (and the audit API with a separate credential), not from the operator's own memory.

| Check | Result | Observation | Evidence |
|---|---|---|---|
| System of record shows the report submitted with the right totals | PASS | system of record matches: {"status": "submitted", "cost_center": "CC-2207", "line_count": 6, "total": 26030, "owner": "priya.sharma"} | evidence/check_01_json.txt |
| Portal page (fresh session) shows submitted report | PASS | observed on /reports/1 in a fresh session | evidence/check_02_web.txt |
| Summary file saved for the employee | PASS | out/expense_summary_1.md contains expected text | evidence/check_03_file.txt |

## Changes made in company systems

- #22 POST http://127.0.0.1:45255/reports with {"title": "Trip - Mumbai - Sep 2026", "cost_center": "CC-2207", "purpose": "Customer workshop - Meridian Retail"} -> **ok**
- #26 POST http://127.0.0.1:45255/reports/1/lines with {"date": "2026-09-14", "category": "Transport", "merchant": "IndiGo Airlines", "amount": "6420", "receipt_ref": "01_indigo_flight", "nights": "1", "note": ""} -> **ok**
- #30 POST http://127.0.0.1:45255/reports/1/lines with {"date": "2026-09-14", "category": "Transport", "merchant": "Meru Cabs", "amount": "1150", "receipt_ref": "02_meru_taxi_airport", "nights": "1", "note": ""} -> **ok**
- #32 POST http://127.0.0.1:45255/reports/1/lines with {"date": "2026-09-14", "category": "Lodging", "merchant": "Taj Lands End, Mumbai", "amount": "15000", "receipt_ref": "03_taj_hotel_folio", "nights": "2", "note": "excluded: Minibar"} -> **ok**
- #34 POST http://127.0.0.1:45255/reports/1/lines with {"date": "2026-09-14", "category": "Meals", "merchant": "Trishna Restaurant", "amount": "1500", "receipt_ref": "04_trishna_dinner", "nights": "1", "note": "excluded: Kingfisher beer x2 (alcohol)"} -> **ok**
- #36 POST http://127.0.0.1:45255/reports/1/lines with {"date": "2026-09-15", "category": "Meals", "merchant": "Kyani & Co.", "amount": "980", "receipt_ref": "05_kyani_lunch", "nights": "1", "note": ""} -> **ok**
- #38 POST http://127.0.0.1:45255/reports/1/lines with {"date": "2026-09-16", "category": "Transport", "merchant": "Uber", "amount": "980", "receipt_ref": "06_uber_return", "nights": "1", "note": ""} -> **ok**
- #41 POST http://127.0.0.1:45255/reports/1/submit with {} -> **ambiguous** (re-checked)
- #44 POST http://127.0.0.1:45255/reports/1/submit with {} -> **ok**
- #46 write file out/expense_summary_1.md (827 chars) -> **ok**

## Human approvals

- #41 `http:POST http://127.0.0.1:45255/reports/1/submit` - approve by scripted-human. Reason it was gated: Submitting an expense report is a financial commitment on behalf of the employee.

## Problems encountered and how they were handled

- Action #27 hit an expired session; the operator signed in again
- Action #41 failed with an unknown outcome; the operator had to re-check state

## Timeline

  1. `tool:update_plan` - ok
  2. `tool:kb_search` - ok
  3. `fs_read:knowledge/expense_reimbursement_procedure.md` - ok
  4. `tool:kb_search` - ok
  5. `fs_read:knowledge/expense_policy.md` - ok
  6. `fs_list:inbox` - ok
  7. `fs_list:inbox/mumbai_trip` - ok
  8. `fs_read:inbox/mumbai_trip/00_travel_request_TR-5521.txt` - ok
  9. `fs_read:inbox/mumbai_trip/01_indigo_flight.txt` - ok
 10. `fs_read:inbox/mumbai_trip/02_meru_taxi_airport.txt` - ok
 11. `fs_read:inbox/mumbai_trip/03_taj_hotel_folio.txt` - ok
 12. `fs_read:inbox/mumbai_trip/04_trishna_dinner.txt` - ok
 13. `fs_read:inbox/mumbai_trip/04b_trishna_dinner_rescan.txt` - ok
 14. `fs_read:inbox/mumbai_trip/05_kyani_lunch.txt` - ok
 15. `fs_read:inbox/mumbai_trip/06_uber_return.txt` - ok
 16. `fs_read:inbox/mumbai_trip/07_fabindia_gifts.txt` - ok
 17. `tool:remember` - ok
 18. `tool:list_systems` - ok
 19. `http:POST http://127.0.0.1:45255/login` - ok HTTP 200
 20. `http:GET http://127.0.0.1:45255/reports` - ok HTTP 200
 21. `http:GET http://127.0.0.1:45255/reports/new` - ok HTTP 200
 22. `http:POST http://127.0.0.1:45255/reports` - ok HTTP 200
 23. `http:GET http://127.0.0.1:45255/reports` - ok HTTP 200
 24. `tool:remember` - ok
 25. `http:GET http://127.0.0.1:45255/reports/1/lines/new` - ok HTTP 200
 26. `http:POST http://127.0.0.1:45255/reports/1/lines` - ok HTTP 200
 27. `http:GET http://127.0.0.1:45255/reports/1/lines/new` - problem HTTP 200
 28. `http:POST http://127.0.0.1:45255/login` - ok HTTP 200
 29. `http:GET http://127.0.0.1:45255/reports/1/lines/new` - ok HTTP 200
 30. `http:POST http://127.0.0.1:45255/reports/1/lines` - ok HTTP 200
 31. `http:GET http://127.0.0.1:45255/reports/1/lines/new` - ok HTTP 200
 32. `http:POST http://127.0.0.1:45255/reports/1/lines` - ok HTTP 200
 33. `http:GET http://127.0.0.1:45255/reports/1/lines/new` - ok HTTP 200
 34. `http:POST http://127.0.0.1:45255/reports/1/lines` - ok HTTP 200
 35. `http:GET http://127.0.0.1:45255/reports/1/lines/new` - ok HTTP 200
 36. `http:POST http://127.0.0.1:45255/reports/1/lines` - ok HTTP 200
 37. `http:GET http://127.0.0.1:45255/reports/1/lines/new` - ok HTTP 200
 38. `http:POST http://127.0.0.1:45255/reports/1/lines` - ok HTTP 200
 39. `http:GET http://127.0.0.1:45255/reports/1` - ok HTTP 200
 40. `http:GET http://127.0.0.1:45255/reports/1` - ok HTTP 200
 41. `http:POST http://127.0.0.1:45255/reports/1/submit` - problem HTTP 503
 42. `http:GET http://127.0.0.1:45255/reports/1` - ok HTTP 200
 43. `http:GET http://127.0.0.1:45255/reports/1` - ok HTTP 200
 44. `http:POST http://127.0.0.1:45255/reports/1/submit` - ok HTTP 200
 45. `http:GET http://127.0.0.1:45255/reports/1` - ok HTTP 200
 46. `fs_write:out/expense_summary_1.md` - ok
 47. `tool:update_plan` - ok

## Final plan state

- [done] Find the procedure and policy
- [done] Work out which trip, read the documents and decide what is claimable
- [done] Create the draft report and add lines in the Expense Portal
- [done] Check the draft, get approval, submit
- [done] Confirm the outcome and write a summary
