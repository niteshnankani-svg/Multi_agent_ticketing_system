# Accuracy + faithfulness: DistilBERT vs Jev

- **Held-out test set**: data/raw/test.csv, 10464 tickets total
- **Stratified sample for matched comparison**: 100 tickets (25/category, random_state=42)
- **Judge**: gemini/gemini-3.8-flash via the Gemini REST API

## Speed (from evals/jev_latency.md and evals/bert_latency.md)

| Classifier | Median | p95 |
|---|---|---|
| BERT | 14.8 ms | 17.9 ms |
| Jev | 354 ms | 449 ms |

## Accuracy (macro-F1 against dataset labels)

- **BERT, full 10464-ticket test set**: 0.896
- **BERT, matched 100-ticket sample**: 0.890
- **Jev, matched 100-ticket sample**: 0.563

## Faithfulness (LLM-judge, same matched sample)

Judges whether the predicted department is a defensible read of the ticket's actual content against the department criteria - independent of the dataset's own (possibly noisy) label.

- **BERT faithful rate**: 73.0% (100 judged)
- **Jev faithful rate**: 93.0% (100 judged)
- **Dataset label judged correct**: 71.0% of sampled tickets (a check on the ground truth itself, not either classifier)

---

## Per-ticket detail (matched sample)

### Disagreements between BERT and Jev (45/100)

- **Dataset label**: `backend` | **BERT**: `backend` (faithful=False) | **Jev**: `general` (faithful=True)
  - Ticket: monitor required tuesday march monitor required hi working client ask monitor board thank lead
  - Judge on BERT: The ticket requests a monitor or monitoring board rather than reporting a product bug, crash, outage, or technical performance issue.
  - Judge on Jev: Because the garbled request for a monitor does not clearly fit finance, backend technical issues, or the specified internal IT access requests, general is a defensible category.

- **Dataset label**: `backend` | **BERT**: `general` (faithful=False) | **Jev**: `internal` (faithful=True)
  - Ticket: new write permissions for dear please informed granted profile permissions approve expense report pm client report unit write permissions old write permissions forms please make tickets other kind reg
  - Judge on BERT: The ticket is an employee access and permission request across internal tools, making 'internal' the appropriate category rather than 'general'.
  - Judge on Jev: The request concerns granting internal employee write and approval permissions across workplace tools like Confluence and Git, which fits internal IT access management.

- **Dataset label**: `backend` | **BERT**: `backend` (faithful=False) | **Jev**: `general` (faithful=True)
  - Ticket: do not have to miscellaneous february pm fwd ski weekend feedback photos dear colleagues help miscellaneous bot body text had thank date re ski weekend feedback photos la ski trip la networking cine s
  - Judge on BERT: The ticket is about an internal company ski trip, photos, and a feedback survey, containing no technical issues, bugs, or product problems related to backend.
  - Judge on Jev: The text relates to social event feedback and does not fit the criteria for finance, backend, or IT internal access requests, making general the appropriate category.

- **Dataset label**: `backend` | **BERT**: `backend` (faithful=False) | **Jev**: `internal` (faithful=True)
  - Ticket: access card deactivated tuesday card deactivated hi please card activate again almost months since assignment client card number name located thank kind regards
  - Judge on BERT: The ticket concerns a deactivated physical access card or badge, which is not a technical or backend software issue.
  - Judge on Jev: Requesting reactivation of an access card directly falls under badge access within the internal department criteria.

- **Dataset label**: `backend` | **BERT**: `backend` (faithful=False) | **Jev**: `internal` (faithful=True)
  - Ticket: meetings canceling meeting room fuji meetings canceling hi colleague has left help cancel meetings booked by him easy estimations room fuji once every weeks easy demo retro room fuji once every weeks 
  - Judge on BERT: The ticket describes an internal administrative request to cancel meeting room reservations made by a departed employee, not a technical bug, crash, or product outage.
  - Judge on Jev: The request concerns internal employee workplace management and administrative assistance for room bookings following a colleague's departure.

- **Dataset label**: `backend` | **BERT**: `backend` (faithful=True) | **Jev**: `general` (faithful=True)
  - Ticket: replacement keyboard replacement keyboard hi replaced keyboard was given few months ago old one continually had issues with keystrokes occasionally sticking control key permanently led software crashe
  - Judge on BERT: The ticket explicitly cites technical malfunctions and software crashes caused by the keyboard issue, making backend a defensible classification under technical issues and crashes.
  - Judge on Jev: Because an employee hardware replacement request does not cleanly fit the specified criteria for finance, backend, or internal, routing to general is defensible.

- **Dataset label**: `backend` | **BERT**: `backend` (faithful=True) | **Jev**: `internal` (faithful=True)
  - Ticket: issues with issues with hi please request for your help with below issue cannot connect when try use laptop outside however can connect programs normal if can you please help fixing able connect next 
  - Judge on BERT: Backend is defensible because the issue involves technical network connectivity and login problems.
  - Judge on Jev: Internal is faithful because the ticket describes an employee needing help connecting their work laptop remotely, which falls under internal IT and VPN support.

- **Dataset label**: `backend` | **BERT**: `backend` (faithful=False) | **Jev**: `internal` (faithful=True)
  - Ticket: internal audit visitor cards review tuesday december visitor cards review hi please each location where visitors cards make information accurate please let location after moved thanks
  - Judge on BERT: The ticket concerns visitor badge audits and physical location tracking, which does not involve technical bugs, product issues, or backend systems.
  - Judge on Jev: The request relates to visitor card/badge reviews across office locations, aligning directly with internal employee and badge access requests.

- **Dataset label**: `backend` | **BERT**: `backend` (faithful=False) | **Jev**: `internal` (faithful=True)
  - Ticket: check laptop admin rights wednesday november pm assigned hello please advised number assigned equipment sd assigned writing because needed install members because assigned install working credentials 
  - Judge on BERT: Laptop admin rights and software installation permissions on internal equipment are employee IT requests rather than backend product issues.
  - Judge on Jev: Requesting laptop administrative rights to install software on company equipment is a standard internal employee IT request.

- **Dataset label**: `backend` | **BERT**: `backend` (faithful=False) | **Jev**: `general` (faithful=True)
  - Ticket: backpack thursday backpack dear please provide backpack thanks engineer
  - Judge on BERT: The ticket contains nonsensical text mentioning backpacks and engineers with no reference to technical product bugs, outages, or login issues.
  - Judge on Jev: Because the text is nonsensical and does not fit finance, technical backend issues, or internal IT requests, general is the most appropriate fallback category.

- **Dataset label**: `backend` | **BERT**: `backend` (faithful=True) | **Jev**: `internal` (faithful=True)
  - Ticket: client portal testing environments wednesday awaiting response client testing environments importance high hello please attach form automatically closed after days response user days after hold awaiti
  - Judge on BERT: The ticket concerns technical configuration, application ports, and testing environment provisioning, which defensibly aligns with backend technical operations.
  - Judge on Jev: The ticket represents an internal staff coordination workflow requesting physical machines and internal testing environment setups.

- **Dataset label**: `finance` | **BERT**: `finance` (faithful=True) | **Jev**: `general` (faithful=True)
  - Ticket: What can you use the disposable cards for?
  - Judge on BERT: Disposable cards are payment instruments, making inquiries about their usage directly related to payments under finance.
  - Judge on Jev: The ticket is a broad informational FAQ about product functionality, making general a reasonable classification.

- **Dataset label**: `finance` | **BERT**: `finance` (faithful=True) | **Jev**: `backend` (faithful=False)
  - Ticket: new purchase po thursday purchase po dear purchased mini cable please log installation kind regards administrator
  - Judge on BERT: The ticket repeatedly references a purchase order (PO) and a new purchase, which aligns directly with the finance department's scope covering sales, billing, and purchasing.
  - Judge on Jev: The request discusses a purchase order rather than a product defect, bug, outage, or technical performance issue suited for backend.

- **Dataset label**: `finance` | **BERT**: `finance` (faithful=True) | **Jev**: `backend` (faithful=True)
  - Ticket: I was removing a dollar amount from my account and now don't have access to my card.
  - Judge on BERT: The issue involves removing funds from an account and managing card access, which falls under financial account and payment methods management.
  - Judge on Jev: The sudden loss of card access following an action can be viewed as an account access issue or technical bug.

- **Dataset label**: `finance` | **BERT**: `finance` (faithful=False) | **Jev**: `internal` (faithful=True)
  - Ticket: printer installation printer installation dear please log assign main printer thanks kind regards engineer
  - Judge on BERT: The request is for local IT printer setup and has nothing to do with billing, invoices, or financial transactions.
  - Judge on Jev: A printer installation request from an engineer fits standard internal workplace IT assistance alongside requests like software licensing and drive access.

- **Dataset label**: `finance` | **BERT**: `finance` (faithful=False) | **Jev**: `general` (faithful=True)
  - Ticket: Is my PIN the same thing as my passcode?
  - Judge on BERT: The query asks for clarification between a PIN and a passcode with no mention of billing, payments, or financial transactions.
  - Judge on Jev: This is a general informational question about terminology that does not clearly fit into finance, backend technical issues, or internal employee support.

- **Dataset label**: `general` | **BERT**: `general` (faithful=False) | **Jev**: `backend` (faithful=True)
  - Ticket: Request for Detailed Documentation on Integrating Pinnacle Studio 24 SaaS Tools. I am seeking comprehensive documentation for integrating Pinnacle Studio 24 SaaS project management tools. Could you pr
  - Judge on BERT: The ticket specifically requests API and integration guidance, which directly falls under the backend department's explicit scope for integrations.
  - Judge on Jev: The request asks for API documentation and instructions for integrating SaaS tools, which directly matches the backend criteria covering integrations.

- **Dataset label**: `general` | **BERT**: `internal` (faithful=False) | **Jev**: `general` (faithful=True)
  - Ticket: forecast reports new pas october pm forecast reports dear please let way extract report contain forecast moment extract report forecast by together report similar analysis report old let questions tha
  - Judge on BERT: The ticket is asking how to extract a forecast report, which does not match the internal IT criteria of drive access, software licenses, VPN, or badges.
  - Judge on Jev: A general how-to inquiry regarding report extraction fits best under general as it does not match the specific criteria for finance, backend, or internal.

- **Dataset label**: `general` | **BERT**: `general` (faithful=True) | **Jev**: `internal` (faithful=True)
  - Ticket: Query for Updating Digital Marketing Tools. Dear Customer Support, I am contacting you to ask about the possibility of upgrading our digital marketing tools on our workstations. We are currently using
  - Judge on BERT: The ticket is a broad inquiry sent to customer support about software upgrade options, costs, and benefits, making general a defensible classification.
  - Judge on Jev: The request involves updating software tools on employee workstations, which can reasonably be interpreted as an internal software licensing and tooling request.

- **Dataset label**: `general` | **BERT**: `general` (faithful=False) | **Jev**: `finance` (faithful=True)
  - Ticket: temporary project code activation oracle november cod hello please code order validate invoice after validation done code closed thank accountant
  - Judge on BERT: The ticket specifically concerns validating an invoice sent to an accountant, which directly falls under finance criteria rather than general.
  - Judge on Jev: The request is from an accountant asking for temporary code activation to validate an invoice, making finance the most defensible category.

- **Dataset label**: `general` | **BERT**: `general` (faithful=False) | **Jev**: `internal` (faithful=True)
  - Ticket: project access request access request hello please help solve issue below regards applications discipline lead sent monday re access request hello pare sa sorry your password incorrect please try agai
  - Judge on BERT: The ticket is an internal IT employee request regarding project access permissions, which directly fits the internal department rather than general.
  - Judge on Jev: The request concerns internal employee project access and permissions, aligning well with the criteria for the internal department.

- **Dataset label**: `general` | **BERT**: `general` (faithful=False) | **Jev**: `backend` (faithful=False)
  - Ticket: change pm oracle tuesday pm tools dear kindly ask write under give him write also he needs write level reports write pipeline codes yr deal client engagement ensure he has write pipeline reports clien
  - Judge on BERT: The ticket is an employee access and permissions request for internal systems like Oracle and PM tools, which belongs under the internal category rather than general.
  - Judge on Jev: The ticket is requesting user write access rather than reporting a product bug, crash, outage, or technical backend issue.

- **Dataset label**: `general` | **BERT**: `general` (faithful=True) | **Jev**: `backend` (faithful=False)
  - Ticket: . Customer Support, seeking guidance on digital strategies for promoting integration between SAP ERP and Shopify. Would appreciate detailed information on methods used to promote these platforms. Look
  - Judge on BERT: The ticket asks for digital marketing and promotional strategies rather than addressing a technical issue, making it a general inquiry.
  - Judge on Jev: While the ticket mentions integration platforms, it seeks promotional and marketing advice rather than reporting a product or technical issue.

- **Dataset label**: `general` | **BERT**: `backend` (faithful=False) | **Jev**: `internal` (faithful=True)
  - Ticket: permission on sent thursday re we updated need permission access hello please know have problem with accession again kind regards sent we updated need permission access ticket resolved fellow we think
  - Judge on BERT: The ticket is an access/permission request rather than a technical outage, bug, or product-level technical issue fitting backend criteria.
  - Judge on Jev: Requests to restore permission and access are standard internal IT support requests.

- **Dataset label**: `general` | **BERT**: `general` (faithful=False) | **Jev**: `backend` (faithful=True)
  - Ticket: resolution not working for project thursday february resolution working hi change manually resolution anymore please take over thank best regards senior engineer
  - Judge on BERT: The ticket describes a technical bug where changing resolution is not working, which directly fits the backend criteria rather than general.
  - Judge on Jev: The request reports a product malfunction regarding resolution settings, making it a clear technical issue appropriate for backend.

- **Dataset label**: `general` | **BERT**: `general` (faithful=False) | **Jev**: `backend` (faithful=True)
  - Ticket: report change request suppliers report re report change suppliers report hi please thanks thursday november pm report change suppliers report hi add columns custom report suppliers report shown image 
  - Judge on BERT: The ticket involves technical database querying, SQL joins, and report schema customization, which falls under technical issues rather than general inquiries.
  - Judge on Jev: The ticket is a technical request regarding database tables, joins, and SQL query troubleshooting for a custom report, which fits backend technical issues.

- **Dataset label**: `general` | **BERT**: `general` (faithful=True) | **Jev**: `internal` (faithful=True)
  - Ticket: access for a new project setup november setup hello kindly ask give which users give client side working name client thanks lot
  - Judge on BERT: The request is garbled and primarily asks a general procedural question about project setup and user assignments rather than a standard internal IT access request.
  - Judge on Jev: The ticket explicitly requests project setup access for users, which can be reasonably interpreted as an internal access provisioning request.

- **Dataset label**: `general` | **BERT**: `general` (faithful=False) | **Jev**: `internal` (faithful=True)
  - Ticket: change of job title oracle july change job title hello grateful make change job title reports details follows job title owner thanks head technical
  - Judge on BERT: This is an internal employee request regarding HR system updates in Oracle, so routing it to general ignores the internal classification.
  - Judge on Jev: The ticket concerns an internal employee profile and system update in Oracle, which defensibly fits internal employee requests.

- **Dataset label**: `general` | **BERT**: `general` (faithful=False) | **Jev**: `backend` (faithful=True)
  - Ticket: pas known pipeline issues known pipeline issues hi we had ticket for for importing opportunity into known pipeline but unfortunately editable can you please check regards officer
  - Judge on BERT: The ticket reports a technical problem importing an opportunity into a pipeline, which fits the backend criteria rather than general.
  - Judge on Jev: The issue involves a software or technical malfunction with pipeline data import, aligning with backend criteria.

- **Dataset label**: `general` | **BERT**: `internal` (faithful=True) | **Jev**: `general` (faithful=True)
  - Ticket: update december days for pas update december days hi kindly ask update number days december according timecard registered thank officer
  - Judge on BERT: The ticket involves an employee requesting an adjustment to their timecard days, which aligns well with internal employee administrative requests.
  - Judge on Jev: Because timecard adjustments are not explicitly covered by the IT-focused examples in the internal department criteria, falling back to general is defensible.

- **Dataset label**: `general` | **BERT**: `general` (faithful=False) | **Jev**: `backend` (faithful=True)
  - Ticket: oracle delete project code delete code hello please delete code attached code opened error thanks
  - Judge on BERT: The ticket concerns an Oracle system error and code/project deletion, which represents a technical issue rather than a general inquiry.
  - Judge on Jev: Routing to backend is appropriate as the request involves technical issues regarding Oracle, code, and an encountered error.

- **Dataset label**: `general` | **BERT**: `general` (faithful=False) | **Jev**: `backend` (faithful=True)
  - Ticket: not generating service now hello could you please investigate why generate sent were sent directly thank you
  - Judge on BERT: The ticket requests an investigation into a technical generation/sending issue, which clearly fits the backend criteria for bugs or technical problems rather than general.
  - Judge on Jev: The user is reporting an unexpected system behavior regarding generating and sending items, which constitutes a technical issue or bug appropriate for backend.

- **Dataset label**: `internal` | **BERT**: `internal` (faithful=True) | **Jev**: `backend` (faithful=True)
  - Ticket: cannot submit the time oracle sent november cannot submit hello have troubles with submitting week trying submit week could you please have look issue added kind regards tester
  - Judge on BERT: Submitting timesheets in Oracle is an internal employee workflow, making internal a defensible department for this request.
  - Judge on Jev: The user is reporting a technical issue or bug preventing submission, which fits the backend criteria for technical problems and bugs.

- **Dataset label**: `internal` | **BERT**: `internal` (faithful=True) | **Jev**: `backend` (faithful=True)
  - Ticket: oracle app error wednesday july app error hi please unlock app period thank senior analyst
  - Judge on BERT: The ticket is an internal employee request from a senior analyst requesting administrative application access or an accounting period unlock.
  - Judge on Jev: The ticket repeatedly cites an 'app error', which defensibly aligns with backend product and technical issue criteria.

- **Dataset label**: `internal` | **BERT**: `internal` (faithful=True) | **Jev**: `backend` (faithful=False)
  - Ticket: activation printer functions activation printer functions hello help activation function printer floor thank
  - Judge on BERT: Requests regarding office hardware like floor printer access and activation are standard internal employee IT support requests.
  - Judge on Jev: The ticket concerns on-premise office equipment rather than product-level software bugs, outages, or backend technical issues.

- **Dataset label**: `internal` | **BERT**: `internal` (faithful=False) | **Jev**: `backend` (faithful=True)
  - Ticket: error when submitting journals error when submitting journals hi could you please help with below keep receiving message when try submit journal something because morning function worked thank you acc
  - Judge on BERT: The issue is a software error/bug rather than an internal access request like shared drives, VPN, badge, or license provisioning.
  - Judge on Jev: The ticket reports an error message and functional failure when submitting journals, which aligns with backend technical issues and bugs.

- **Dataset label**: `internal` | **BERT**: `general` (faithful=False) | **Jev**: `backend` (faithful=True)
  - Ticket: cancel remove cache request oracle thursday february cancel dear please cancel delete cannot myself withdraw option seems deactivated kind regards technical consultant
  - Judge on BERT: The ticket concerns a technical Oracle system request and a deactivated feature, making it a technical issue rather than a general inquiry.
  - Judge on Jev: The request involves a technical issue regarding an Oracle cache operation and an option that appears deactivated, which aligns with backend technical support.

- **Dataset label**: `internal` | **BERT**: `general` (faithful=True) | **Jev**: `internal` (faithful=True)
  - Ticket: time sheet sheet you can line for was done why have approve his manager frankfurt main frankfurt main
  - Judge on BERT: The text is largely garbled and timesheet approvals do not strictly match the specific IT access examples listed under internal, making general a defensible catch-all choice.
  - Judge on Jev: The mention of timesheets and manager approval points to internal employee operations, making internal a reasonable classification.

- **Dataset label**: `internal` | **BERT**: `internal` (faithful=True) | **Jev**: `backend` (faithful=True)
  - Ticket: unable to complete with less than hours per day unable complete with less than hours per hi having problems with allowing log normal hours per save or progress any days with less than hours logged per
  - Judge on BERT: The issue involves an internal employee reporting problems with an internal time-tracking system, making internal support a defensible routing.
  - Judge on Jev: The ticket explicitly describes a software bug or glitch preventing progress following an upgrade, which fits the backend criteria for technical issues and bugs.

- **Dataset label**: `internal` | **BERT**: `internal` (faithful=True) | **Jev**: `backend` (faithful=False)
  - Ticket: oracle problem with absence record problem absence record dear submit special leave reason patron because date past help absence record best regards senior developer
  - Judge on BERT: The ticket concerns an internal employee dealing with an HR leave record in an internal system (Oracle), making 'internal' a faithful classification.
  - Judge on Jev: The issue involves internal HR leave management rather than a technical bug or outage affecting the company's external-facing product.

- **Dataset label**: `internal` | **BERT**: `internal` (faithful=True) | **Jev**: `finance` (faithful=False)
  - Ticket: ad oracle sent monday ad importance high hello have access but can category attached you print screen thank you accountant
  - Judge on BERT: The ticket concerns an internal employee (an accountant) dealing with Oracle system access permissions, which aligns well with internal IT access requests.
  - Judge on Jev: The ticket involves an access issue rather than billing, invoices, payments, or refunds, making finance inappropriate despite the sender being an accountant.

- **Dataset label**: `internal` | **BERT**: `internal` (faithful=True) | **Jev**: `backend` (faithful=True)
  - Ticket: error days sent monday hire error days hi displaying only days correct believe can you please advise correct number can make adjustment regards manager nj
  - Judge on BERT: The ticket concerns an internal manager's request regarding new hire days and records adjustments, fitting an internal employee/HR operational workflow.
  - Judge on Jev: The ticket references an error with the system displaying the wrong number of days, which can defensibly be interpreted as a technical or product bug under backend.

- **Dataset label**: `internal` | **BERT**: `backend` (faithful=False) | **Jev**: `general` (faithful=True)
  - Ticket: open day wednesday hello guys please thank lead wednesday maine si cafeteria care ex diverse la
  - Judge on BERT: The ticket contains no mention of technical bugs, outages, crashes, or backend product issues.
  - Judge on Jev: The text is an unclear or garbled message about an open day and cafeteria that does not fit the specific criteria for finance, backend, or internal IT requests.

- **Dataset label**: `internal` | **BERT**: `backend` (faithful=False) | **Jev**: `internal` (faithful=True)
  - Ticket: secured space access tuesday july secured hi please provide secured area she based working today best regards friday july pm leads re secured please also allow members believe done same best regards j
  - Judge on BERT: The request is regarding physical badge access and room permissions for employees in a secured office space, not a technical bug or product issue.
  - Judge on Jev: The ticket explicitly requests entrance cards and physical access to secured office areas for internal employees, which directly aligns with internal badge access criteria.

- **Dataset label**: `internal` | **BERT**: `internal` (faithful=True) | **Jev**: `finance` (faithful=True)
  - Ticket: overtime overtime sorry think wrong october re overtime hello please accept apologies give wrong steer think gone cheers change control presence mob please annual leave october overtime hi checking oc
  - Judge on BERT: The ticket is an internal employee request regarding overtime and annual leave compensation.
  - Judge on Jev: The ticket inquires about being paid for October overtime, which relates directly to payments.
