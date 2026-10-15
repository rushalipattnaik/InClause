# Knowledge base review checklist

Every note in `legal_data/knowledge_base.json` starts with `"reviewed": false`, and the app labels
such notes "not yet reviewed". Check each note against the primary source, fix the wording if needed,
then set `"reviewed": true` for that entry. The search index rebuilds automatically.

Primary sources: Indian Acts at https://www.indiacode.nic.in , RBI directions at https://www.rbi.org.in ,
the Digital Personal Data Protection Act, 2023 on the same India Code site.

## A. Notes that cite a law or regulation (12): verify the section number AND what it says

- [ ] **indemnity**: Indemnity clauses
  - Source line: General legal concept; Indian Contract Act, 1872, s.124 (summary)
  - Text: An indemnity clause makes one party promise to cover the other party's losses, damages or legal costs. Check whether it is mutual, whether it has a cap, and whether it covers losses the indemnifying party did not cause. Under the Indian Contract Act, 1872 (s.124), a contract of indemnity is a promise to save the other party from loss caused by the promisor's own conduct or by the conduct of any other person.

- [ ] **arbitration**: Arbitration and dispute resolution
  - Source line: General legal concept; Arbitration and Conciliation Act, 1996 (summary)
  - Text: An arbitration clause sends disputes to a private arbitrator instead of a court. It can be faster but usually limits appeals. In India arbitration is governed by the Arbitration and Conciliation Act, 1996, and Indian courts have set limits on one party unilaterally choosing the sole arbitrator. Waivers of class-action rights are common in consumer terms from other countries.

- [ ] **lease-eviction**: Lease termination and eviction
  - Source line: General legal concept; Transfer of Property Act, 1882, s.106 (summary)
  - Text: A tenant normally gets notice before a landlord can end a tenancy, and rent control laws differ by state. Under the Transfer of Property Act, 1882 (s.106), unless the contract or local law says otherwise, a month-to-month lease needs fifteen days' notice and a year-to-year lease for farming or manufacturing needs six months' notice. A fixed-term lease, such as eleven months, ends when the term expires.

- [ ] **data-consent**: Personal data and consent
  - Source line: General legal concept; Digital Personal Data Protection Act, 2023 (summary)
  - Text: Clauses on personal data say what is collected, why, who it is shared with, and how long it is kept. Marketing and third-party sharing are usually separate purposes that need clear consent. India's Digital Personal Data Protection Act, 2023 requires free, specific and informed consent for processing personal data.

- [ ] **liquidated-damages-in**: Liquidated damages and penalties (India)
  - Source line: Indian Contract Act, 1872, s.74 (summary)
  - Text: Under s.74 of the Indian Contract Act, 1872, when a contract names a sum payable on breach or imposes a penalty, the injured party can claim reasonable compensation not exceeding that named amount, whether or not actual loss is proved. Courts therefore do not simply enforce a very large penalty as written.

- [ ] **restraint-of-trade-in**: Restraint of trade (India)
  - Source line: Indian Contract Act, 1872, s.27 (summary)
  - Text: Section 27 of the Indian Contract Act, 1872 says an agreement that restrains someone from carrying on a lawful profession, trade or business is void to that extent, with a narrow exception for the sale of goodwill. Restrictions that apply after employment ends are therefore generally hard to enforce in India, while duties during employment are treated differently.

- [ ] **restraint-of-legal-proceedings-in**: Restraint of legal proceedings (India)
  - Source line: Indian Contract Act, 1872, s.28 (summary)
  - Text: Section 28 of the Indian Contract Act, 1872 makes void an agreement that completely stops a party from going to the ordinary courts or that cuts short the time to sue. Referring disputes to arbitration is an exception. A clause saying a party can never sue is therefore likely to be unenforceable.

- [ ] **breach-compensation-in**: Compensation for breach (India)
  - Source line: Indian Contract Act, 1872, s.73 (summary)
  - Text: Section 73 of the Indian Contract Act, 1872 says a party who suffers loss because the other broke the contract can claim compensation for loss that naturally arose from the breach or that both parties knew was likely. It does not cover remote or indirect loss.

- [ ] **frustration-in**: Impossibility of performance (India)
  - Source line: Indian Contract Act, 1872, s.56 (summary)
  - Text: Section 56 of the Indian Contract Act, 1872 says an agreement to do an impossible act is void, and a contract becomes void when the act later becomes impossible or unlawful through an event the parties could not prevent. This works alongside force majeure clauses.

- [ ] **loan-prepayment**: Prepayment and foreclosure charges
  - Source line: General legal concept; RBI directions (verify the current version)
  - Text: Prepayment or foreclosure charges are fees for repaying a loan early. RBI directions restrict such charges on many floating-rate loans to individual borrowers, so check the current rules for your loan type. Processing fees are often kept even if the loan is not taken.

- [ ] **personal-guarantee**: Guarantees
  - Source line: General legal concept; Indian Contract Act, 1872, s.126-128 (summary)
  - Text: In a guarantee, a person promises to pay if the borrower does not. Under s.128 of the Indian Contract Act, 1872, the guarantor's liability is generally the same as the borrower's unless the contract says otherwise. Guarantees that also cover future borrowings can create open-ended exposure.

- [ ] **unfair-contract-terms-in**: Unfair contract terms for consumers (India)
  - Source line: Consumer Protection Act, 2019, s.2(46) (summary)
  - Text: The Consumer Protection Act, 2019 (s.2(46)) defines an unfair contract between a trader or service provider and a consumer as one with terms that significantly change the consumer's rights, such as manifestly excessive security deposits, penalties out of proportion to the loss, refusing early repayment, one-sided termination without reasonable cause, or unreasonable charges. It applies to consumer contracts, not business-to-business ones.

## B. General-concept notes (25): check that the wording is accurate and not misleading

- [ ] **termination-notice**: Termination and notice
  - Text: Termination clauses say how and when a contract can end. Look at the notice period, whether a reason is needed ('for cause') or not ('for convenience'), and whether both parties have the same right. A clause letting only one party end the contract at any time without notice is heavily one-sided.

- [ ] **limitation-of-liability**: Limitation of liability
  - Text: A limitation of liability clause caps how much one party must pay if something goes wrong, or excludes certain losses such as lost profit or indirect damages. A low cap can leave the other party with little real remedy. Many legal systems do not allow a party to exclude liability for its own fraud or wilful misconduct.

- [ ] **confidentiality**: Confidentiality and NDAs
  - Text: Confidentiality clauses restrict sharing of non-public information. Check what counts as confidential, the exceptions (such as information already public or required by law), and how long the duty lasts after the contract ends. Breaching it can carry separate damages.

- [ ] **non-compete**: Non-compete clauses
  - Text: A non-compete restricts someone from working for or starting a competing business for a set time and area. How enforceable it is depends on the legal system, and broad restrictions in terms of time, place or activity are the most likely to be challenged.

- [ ] **force-majeure**: Force majeure
  - Text: A force majeure clause excuses a party from performing when extraordinary events outside its control, such as floods, war, strikes or government orders, make performance impossible or impractical. Only events listed or fairly described are covered, and payment obligations are often excluded.

- [ ] **auto-renewal**: Automatic renewal
  - Text: An automatic renewal clause extends the contract for another full term unless a party cancels before a deadline. A long cancellation window or an easy-to-miss deadline can lock a person into another term, so the exact date and method of cancelling matter.

- [ ] **late-payment**: Late payment fees and penalties
  - Text: Late payment clauses add a fee or interest when money is paid after the due date. Check whether the charge is a flat fee or grows daily or monthly, when the grace period ends, and whether the amount is reasonable compared to the actual loss.

- [ ] **security-deposit**: Security deposits
  - Text: A security deposit is held as protection against unpaid dues or damage. A fair clause says what can be deducted, requires deductions to be justified, and gives a time limit for returning the balance. A deposit declared 'non-refundable' or forfeited for any reason is unusual and one-sided.

- [ ] **waiver-of-rights**: Waiver of rights
  - Text: A waiver clause says a party gives up a right it would otherwise have, for example the right to sue, claim set-off, or object to later changes. Waivers are often buried in dense text but can remove important protections, so check exactly which right is given up.

- [ ] **governing-law**: Governing law and jurisdiction
  - Text: A governing law clause chooses which country's or state's law applies, and a jurisdiction clause chooses which courts hear disputes. An exclusive court in a distant city can make it costly for one party to enforce its rights.

- [ ] **unilateral-changes**: One-sided changes to the contract
  - Text: Clauses that let one party change prices, terms, duties or the rate of interest at its own discretion, or treat continued use as acceptance, give that party power over the deal. A fair version requires advance notice and a right to exit without penalty.

- [ ] **non-refundable**: Non-refundable payments and forfeiture
  - Text: A non-refundable or forfeited payment means money is kept even when the other side does not perform or the deal is cancelled. Courts may look at whether the amount is reasonable and not a disguised penalty. Check which payments are covered and in which situations.

- [ ] **ip-ownership**: Intellectual property ownership and licences
  - Text: These clauses decide who owns work, inventions or content created under the contract and what licence the other party gets. Watch for ownership of work done outside work hours, ownership transferred before payment, and licences that are perpetual, irrevocable or worldwide.

- [ ] **interest-charges**: Interest and compounding
  - Text: Interest clauses set the rate and how it is calculated. Check the annual rate, whether it compounds monthly, whether a higher 'penal' rate applies on default, and whether the lender can change the rate on its own.

- [ ] **warranty-disclaimer**: Warranty disclaimers
  - Text: A warranty disclaimer says goods or services are provided 'as is' with no promise about quality or fitness. It shifts the risk of defects to the buyer, and consumer protection laws may limit how far such a disclaimer applies.

- [ ] **assignment**: Assignment of the contract
  - Text: An assignment clause controls whether a party can transfer its rights or duties to someone else. A fair clause needs consent or notice. Letting only one party transfer the contract can leave the other dealing with a stranger.

- [ ] **entire-agreement**: Entire agreement
  - Text: An entire agreement clause says the written contract replaces all earlier talks and promises. Anything promised verbally or by email that matters should therefore be written into the contract before signing.

- [ ] **severability**: Severability
  - Text: A severability clause says that if one part of the contract is found invalid, the rest stays in force. It is common and generally neutral.

- [ ] **notices**: Notices
  - Text: A notice clause says how formal communications must be sent and when they count as received. Check the address or email, the deemed delivery time, and whether a missed notice could cost you a right or deadline.

- [ ] **employment-notice-period**: Employment notice period
  - Text: A notice period is how long an employee or employer must give before ending employment. Long periods, recovery of salary if the notice is not served, or a notice that only the employee must give can make leaving costly.

- [ ] **rent-escalation**: Rent increases
  - Text: A rent escalation clause raises rent at set intervals. Check the percentage, how often it applies, and whether it compounds, because a yearly percentage increase adds up over a long lease.

- [ ] **exclusivity**: Exclusivity
  - Text: An exclusivity clause stops a party from working with others or buying elsewhere. Check how long it lasts, what is covered, and whether the other side gives anything in return.

- [ ] **subletting**: Subletting
  - Text: A subletting clause controls whether a tenant can hand over the premises to someone else. Most leases ban it without the landlord's written consent, and breaking this rule can be a ground to end the lease.

- [ ] **repairs-maintenance**: Repairs and maintenance in leases
  - Text: This clause divides repair costs between landlord and tenant. Tenants commonly cover small day-to-day repairs while the landlord covers structure and major systems. A clause making the tenant pay for all repairs, whatever the cause, shifts risk heavily.

- [ ] **payment-terms**: Payment terms
  - Text: Payment clauses set amounts, due dates, and the method of payment. Check whether payment depends on the other side's satisfaction, whether it can be withheld, and what happens if you pay late.

## When you are done

Entries you could not verify should be reworded to say less, or deleted. A shorter, correct
knowledge base is better than a longer one with claims you cannot defend.
