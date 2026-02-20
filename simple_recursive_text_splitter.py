import re

class SimpleRecursiveSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Priority: Paragraphs -> Newlines -> Sentences -> Words -> Characters
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text):
        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text, separators):
        final_chunks = []
        
        # 1. Determine which separator to use
        separator = separators[-1]
        new_separators = []
        for i, s in enumerate(separators):
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                new_separators = separators[i + 1:]
                break

        # 2. Split the text
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # 3. Merge splits into chunks
        good_splits = []
        for s in splits:
            # If the individual split is too large, recurse down to the next separator
            if len(s) > self.chunk_size:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                
                if not new_separators:
                    final_chunks.append(s) # Nowhere left to split
                else:
                    recursive_chunks = self._recursive_split(s, new_separators)
                    final_chunks.extend(recursive_chunks)
            else:
                good_splits.append(s)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _merge_splits(self, splits, separator):
        chunks = []
        current_doc = []
        total_len = 0
        
        for s in splits:
            _len = len(s)
            # If adding this split exceeds chunk_size
            if total_len + _len + (len(separator) if current_doc else 0) > self.chunk_size:
                if total_len > 0:
                    doc = separator.join(current_doc)
                    chunks.append(doc)
                    
                    # Keep track of overlap: walk backwards from the end of current_doc
                    # to keep some context for the next chunk
                    while total_len > self.chunk_overlap or (total_len > 0 and len(current_doc) > 1):
                        popped = current_doc.pop(0)
                        total_len -= (len(popped) + len(separator))

            current_doc.append(s)
            total_len += _len + (len(separator) if len(current_doc) > 1 else 0)

        if current_doc:
            chunks.append(separator.join(current_doc))
        
        return chunks

# --- Example Usage ---

if __name__ == "__main__":
    text = """# A Republican plan to overhaul voting is back. Here's what's new in the bill
    A Republican voting overhaul is back on Capitol Hill — with an added photo identification provision and an altered name — as President Trump seeks to upend elections in a midterm year. Opponents say the legislation would disenfranchise millions of voters.

The Safeguard American Voter Eligibility Act — now dubbed the SAVE America Act — narrowly passed the U.S. House last week, with all Republicans and one Democrat backing the bill.

Its approval came about 10 months after House Republicans last passed the SAVE Act.

The measure, which would transform voter registration and voting across the country, faces persistent hurdles in the GOP-led Senate due to Democratic disapproval and the 60-vote threshold to clear the legislative filibuster. Some Republicans have called for maneuvering around the filibuster to pass the legislation, but GOP leadership has been cool to the idea.

A Vote Here sign is posted amongst political signs as people arrive to vote at the Rutherford County Annex Building, an early voting site, Oct. 17, 2024, in Rutherfordton, N.C.
Politics
House GOP pushes strict proof-of-citizenship requirement for voters
The overhaul would require eligible voters to provide proof of citizenship — like a valid U.S. passport, or a birth certificate plus valid photo identification — when registering to vote. The new iteration adds a requirement that voters also provide photo ID when casting their ballot.

"This bill takes a strong piece of legislation, the SAVE Act, and makes it even stronger in the SAVE America Act," Rep. Bryan Steil, R-Wis., chair of the Committee on House Administration, said in prepared remarks on Capitol Hill last week.

It's already illegal for non-U.S. citizens to vote in federal elections, and proven instances of fraud — including by noncitizens — are vanishingly rare.

But Steil and other Republicans say current law, which requires sworn attestation of citizenship under penalty of perjury, is not strong enough, and documentary proof is needed.

U.S. Rep. Bryan Steil, R-Wis., speaks during a House Rules Committee meeting about the SAVE America Act on Feb. 10. At right is Rep. Joe Morelle, D-N.Y.
U.S. Rep. Bryan Steil, R-Wis., speaks during a House Rules Committee meeting about the SAVE America Act on Feb. 10. At right is Rep. Joe Morelle, D-N.Y.

Samuel Corum/Getty Images
Some states already take steps to verify citizenship for newly registered voters. And three dozen states also require voters to show an ID to cast a ballot, with some mandating it be a photo ID, while others allow additional options, such as a bank statement.

Democrats and voting rights advocates say the new SAVE Act is even worse than the prior iteration, and that the legislation's two main identification requirements would make voting notably more difficult for tens of millions of Americans who don't have easy access to necessary personal documentation. About half of Americans didn't have a passport as of 2023, for instance.

The measure's provisions would take effect immediately, a prospect that opponents see as placing an unfair burden on voters and election officials right before millions are set to cast midterm ballots, and without extra funding. Those election officials would also face criminal penalties, including imprisonment, for registering voters without proof of citizenship.

The bill's prospects appear slim in the Senate, even as Trump and members of his administration ramp up public messaging in favor of the overhaul — often by pointing to polling that shows 8 in 10 Americans support the proof-of-citizenship and photo ID provisions.

An FBI employee stands inside the Fulton County election hub, near Atlanta, as the FBI executes a search warrant for 2020 election materials, on Jan. 28.
Elections
The FBI seizure of Georgia 2020 election ballots relies on debunked claims
Trump tried to overturn his 2020 election loss and has long railed baselessly about corrupt elections, and many of the president's opponents see the push for the SAVE bill as intertwined with his efforts to raise doubts about voting and interfere with this year's midterms.

Michael Waldman, head of the Brennan Center of Justice, which advocates for expanded voting access, described the measure as "Trump's power grab in legislative garb."

Trump raised alarms by suggesting recently that Republicans should "nationalize" elections, and last week he teased a new executive order, writing on social media: "There will be Voter I.D. for the Midterm Elections, whether approved by Congress or not!"

The U.S. Constitution grants states and Congress control over election rules, and a 2025 executive order from Trump, which sought to require proof of citizenship for voter registration, has been halted by federal judges who say the order's provisions exceed a president's authority.

Here are four new items in the new SAVE Act:
1. The photo ID provision only lists valid U.S. passports, driver's licenses, state IDs, military IDs and tribal IDs as acceptable. Voters who do not present one must vote provisionally and return in three days with an ID, or sign an affidavit that says they have "a religious objection to being photographed."

Notably, people who don't vote in person must also submit a copy of a valid photo ID.

2. The legislation includes new guidelines for name discrepancies in proof-of-citizenship documents. That includes, per the bill, "an affidavit signed by the applicant attesting that the name on the documentation is a previous name of the applicant."

This nods at one criticism of the measure, which is that tens of millions of women changed their name after getting married, and so their current names don't match their birth certificate.

3. The measure provides exemptions for absent service members and their families.

Anthony Nel has been a U.S. citizen for more than a decade and a regular voter for the past nine years, but he was flagged as a potential noncitizen and removed from the voter rolls after he did not respond to a county notice within 30 days.
Elections
Trump's SAVE tool is looking for noncitizen voters. But it's flagging U.S. citizens too
4. The bill requires each state to submit its voter list to the Department of Homeland Security for comparison with the Systematic Alien Verification for Entitlements (SAVE) system.

The Trump administration has overhauled SAVE, turning it into a de facto national citizenship system. Despite privacy and data accuracy concerns, a number of states have eagerly used the new SAVE tool to try to identify and remove noncitizens on their voter rolls. But SAVE has erroneously flagged U.S. citizens too.
"""
    chunk_size = 800
    print(f"Splitting text with {chunk_size}")
    # Set chunk_size to 300 to see it in action (Small enough to force splits)
    splitter = SimpleRecursiveSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = splitter.split_text(text)

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
        print(chunk)
