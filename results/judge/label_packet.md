# Faithfulness labeling packet

50 answers sampled (seed 1234) from `results/runs/hybrid_rerank_8b.records.jsonl`. For each sample, read the answer and the passages it cites.

Label **1** if every claim in the answer is supported by the cited passages, otherwise **0**. Judge support only, not whether the answer is correct for the question. The model's own verdict is not shown.

Write each label after the comma in `results/judge/owner_labels.csv` (one line per sample id), save, then run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\finalize_owner_labels.ps1
```

## S01

**Question:** What type of Turing machine is capable of multiple actions and extends into a variety of computational paths?

**Answer:** A non-deterministic Turing machine [a01p013:0:80903f9f].

**Passage `a01p013:0:80903f9f`** (Computational complexity theory)

> A deterministic Turing machine is the most basic Turing machine, which uses a fixed set of rules to determine its future actions. A probabilistic Turing machine is a deterministic Turing machine with an extra supply of random bits. The ability to make probabilistic decisions often helps algorithms solve problems more efficiently. Algorithms that use random bits are called randomized algorithms. A non-deterministic Turing machine is a deterministic Turing machine with an added feature of non-determinism, which allows a Turing machine to have multiple possible future actions from a given state. One way to view non-determinism is that the Turing machine branches into many possible computational paths at each step, and if it solves the problem in any of these branches, it is said to have solved the problem.

## S02

**Question:** In the determination of complexity classes, what are two examples of types of Turing machines?

**Answer:** Deterministic Turing machines, non-deterministic Turing machines [a01p014:0:46c3bc1c]

**Passage `a01p014:0:46c3bc1c`** (Computational complexity theory)

> Many types of Turing machines are used to define complexity classes, such as deterministic Turing machines, probabilistic Turing machines, non-deterministic Turing machines, quantum Turing machines, symmetric Turing machines and alternating Turing machines. They are all equally powerful in principle, but when resources (such as time or space) are bounded, some of these may be more powerful than others.

## S03

**Question:** What are two examples of measurements are bound within algorithms to establish complexity classes?

**Answer:** Time and space [a01p025:0:53c475ac].

**Passage `a01p025:0:53c475ac`** (Computational complexity theory)

> Many important complexity classes can be defined by bounding the time or space used by the algorithm. Some important complexity classes of decision problems defined in this manner are the following:

## S04

**Question:** IP and AM are most commonly defined by what type of proof system?

**Answer:** Interactive proof systems [a01p026:0:98b27f31]

**Passage `a01p026:0:98b27f31`** (Computational complexity theory)

> Other important complexity classes include BPP, ZPP and RP, which are defined using probabilistic Turing machines; AC and NC, which are defined using Boolean circuits; and BQP and QMA, which are defined using quantum Turing machines. #P is an important complexity class of counting problems (not decision problems). Classes like IP and AM are defined using Interactive proof systems. ALL is the class of all decision problems.

## S05

**Question:** What would create a conflict between a problem X and problem C within the context of reduction?

**Answer:** A problem X would create a conflict with problem C if X is in C but harder than any other problem in C, meaning it's not solvable in polynomial time by any problem in C [a01p032:0:4324d375]. This would make X a complete problem for C, implying X is among the hardest problems in C, and no known polynomial-time solution exists for X within C [a01p032:0:4324d375].

**Passage `a01p032:0:4324d375`** (Computational complexity theory)

> If a problem X is in C and hard for C, then X is said to be complete for C. This means that X is the hardest problem in C. (Since many problems could be equally hard, one might say that X is one of the hardest problems in C.) Thus the class of NP-complete problems contains the most difficult problems in NP, in the sense that they are the ones most likely not to be in P. Because the problem P = NP is not solved, being able to reduce a known NP-complete problem, Π2, to another problem, Π1, would indicate that there is no known polynomial-time solution for Π1. This is because a polynomial-time solution to Π1 would yield a polynomial-time solution to Π2. Similarly, because all NP problems can be reduced to the set, finding an NP-complete problem that can be solved in polynomial time would mean that P = NP.

## S06

**Question:** What class is most commonly not ascribed to the graph isomorphism problem in spite of definitive determination?

**Answer:** NP-complete [a01p036:0:bac7e499]

**Passage `a01p036:0:bac7e499`** (Computational complexity theory)

> The graph isomorphism problem is the computational problem of determining whether two finite graphs are isomorphic. An important unsolved problem in complexity theory is whether the graph isomorphism problem is in P, NP-complete, or NP-intermediate. The answer is not known, but it is believed that the problem is at least not NP-complete. If graph isomorphism is NP-complete, the polynomial time hierarchy collapses to its second level. Since it is widely believed that the polynomial hierarchy does not collapse to any finite level, it is believed that graph isomorphism is not NP-complete. The best algorithm for this problem, due to Laszlo Babai and Eugene Luks has run time 2O(√(n log(n))) for graphs with n vertices.

## S07

**Question:** What is an example of a problem to which effective algorithms have provided a solution in spite of the intractability associated with the breadth of sizes?

**Answer:** The decision problem in Presburger arithmetic [a01p042:0:5a1c9971] is an example of a problem to which effective algorithms have provided a solution in spite of the intractability associated with the breadth of sizes.

**Passage `a01p042:0:5a1c9971`** (Computational complexity theory)

> What intractability means in practice is open to debate. Saying that a problem is not in P does not imply that all large cases of the problem are hard or even that most of them are. For example, the decision problem in Presburger arithmetic has been shown not to be in P, yet algorithms have been written that solve the problem in reasonable times in most cases. Similarly, algorithms can solve the NP-complete knapsack problem over a wide range of sizes in less than quadratic time and SAT solvers routinely handle large instances of the NP-complete Boolean satisfiability problem.

## S08

**Question:** What kind of economy did northern California start to grow in the 2000s?

**Answer:** The economy of northern California in the 2000s became dominated by a tech-oriented sector, as stated in [a02p021:0:02d25619].

**Passage `a02p021:0:02d25619`** (Southern California)

> As of the 2010 United States Census, southern California has a population of 22,680,010. Despite a reputation for high growth rates, southern California's rate grew less than the state average of 10.0% in the 2000s as California's growth became concentrated in the northern part of the state due to a stronger, tech-oriented economy in the Bay Area and an emerging Greater Sacramento region.

## S09

**Question:** Where was the first settlement in Victoria?

**Answer:** Sullivan Bay [a04p006:0:29d3c6fd]

**Passage `a04p006:0:29d3c6fd`** (Victoria (Australia))

> Prior to European settlement, the area now constituting Victoria was inhabited by a large number of Aboriginal peoples, collectively known as the Koori. With Great Britain having claimed the entire Australian continent east of the 135th meridian east in 1788, Victoria was included in the wider colony of New South Wales. The first settlement in the area occurred in 1803 at Sullivan Bay, and much of what is now Victoria was included in the Port Phillip District in 1836, an administrative division of New South Wales. Victoria was officially created a separate colony in 1851, and achieved self-government in 1855. The Victorian gold rush in the 1850s and 1860s significantly increased both the population and wealth of the colony, and by the Federation of Australia in 1901, Melbourne had become the largest city and leading financial centre in Australasia.

## S10

**Question:** How much did the population of Victoria increase in ten years after the discovery of gold?

**Answer:** The population of Victoria increased sevenfold from 76,000 to 540,000 in ten years after the discovery of gold [a04p016:0:fc82035f].

**Passage `a04p016:0:fc82035f`** (Victoria (Australia))

> On 1 July 1851, writs were issued for the election of the first Victorian Legislative Council, and the absolute independence of Victoria from New South Wales was established proclaiming a new Colony of Victoria. Days later, still in 1851 gold was discovered near Ballarat, and subsequently at Bendigo. Later discoveries occurred at many sites across Victoria. This triggered one of the largest gold rushes the world has ever seen. The colony grew rapidly in both population and economic power. In ten years the population of Victoria increased sevenfold from 76,000 to 540,000. All sorts of gold records were produced including the "richest shallow alluvial goldfield in the world" and the largest gold nugget. Victoria produced in the decade 1851–1860 20 million ounces of gold, one third of the world's output[citation needed].

## S11

**Question:** What leader led the Dutch Revolt and wrote Apologie?

**Answer:** William the Silent, who wrote Apologie, led the Dutch Revolt [a05p029:0:123a1470].

**Passage `a05p029:0:123a1470`** (Huguenot)

> Some Huguenots fought in the Low Countries alongside the Dutch against Spain during the first years of the Dutch Revolt (1568–1609). The Dutch Republic rapidly became a destination for Huguenot exiles. Early ties were already visible in the "Apologie" of William the Silent, condemning the Spanish Inquisition, which was written by his court minister, the Huguenot Pierre L'Oyseleur, lord of Villiers. Louise de Coligny, daughter of the murdered Huguenot leader Gaspard de Coligny, married William the Silent, leader of the Dutch (Calvinist) revolt against Spanish (Catholic) rule. As both spoke French in daily life, their court church in the Prinsenhof in Delft held services in French. The practice has continued to the present day. The Prinsenhof is one of the 14 active Walloon churches of the Dutch Reformed Church.

## S12

**Question:** For how long did Huguenots continue to use French names?

**Answer:** Huguenots continued to use French first names and surnames for their children well into the nineteenth century [a05p039:0:ad231892].

**Passage `a05p039:0:ad231892`** (Huguenot)

> Most of the Huguenot congregations (or individuals) in North America eventually affiliated with other Protestant denominations with more numerous members. The Huguenots adapted quickly and often married outside their immediate French communities, which led to their assimilation. Their descendants in many families continued to use French first names and surnames for their children well into the nineteenth century. Assimilated, the French made numerous contributions to United States economic life, especially as merchants and artisans in the late Colonial and early Federal periods. For example, E.I. du Pont, a former student of Lavoisier, established the Eleutherian gunpowder mills.

## S13

**Question:** In what year did Jerónimo de Ayanz y Beaumont patent a water pump for draining mines?

**Answer:** 1606 [a06p008:0:d998c020]

**Passage `a06p008:0:d998c020`** (Steam engine)

> The history of the steam engine stretches back as far as the first century AD; the first recorded rudimentary steam engine being the aeolipile described by Greek mathematician Hero of Alexandria. In the following centuries, the few steam-powered "engines" known were, like the aeolipile, essentially experimental devices used by inventors to demonstrate the properties of steam. A rudimentary steam turbine device was described by Taqi al-Din in 1551 and by Giovanni Branca in 1629. Jerónimo de Ayanz y Beaumont received patents in 1606 for fifty steam powered inventions, including a water pump for draining inundated mines. Denis Papin, a Huguenot refugee, did some useful work on the steam digester in 1679, and first used a piston to raise weights in 1690.

## S14

**Question:** Oxygen is released in cellular respiration by?

**Answer:** Organisms [a07p001:0:0d5d2127]

**Passage `a07p001:0:0d5d2127`** (Oxygen)

> Many major classes of organic molecules in living organisms, such as proteins, nucleic acids, carbohydrates, and fats, contain oxygen, as do the major inorganic compounds that are constituents of animal shells, teeth, and bone. Most of the mass of living organisms is oxygen as it is a part of water, the major constituent of lifeforms. Oxygen is used in cellular respiration and released by photosynthesis, which uses the energy of sunlight to produce oxygen from water. It is too chemically reactive to remain a free element in air without being continuously replenished by the photosynthetic action of living organisms. Another form (allotrope) of oxygen, ozone (O
> 3), strongly absorbs UVB radiation and consequently the high-altitude ozone layer helps protect the biosphere from ultraviolet radiation, but is a pollutant near the surface where it is a by-product of smog. At even higher low earth orbit altitudes, sufficient atomic oxygen is present to cause erosion for spacecraft.

## S15

**Question:** What gave Priestley the claim to being the first discovered of oxygen?

**Answer:** Priestley published his findings first in 1775, in a paper titled "An Account of Further Discoveries in Air" [a07p003:0:8bc53d10].

**Passage `a07p003:0:8bc53d10`** (Oxygen)

> In the meantime, on August 1, 1774, an experiment conducted by the British clergyman Joseph Priestley focused sunlight on mercuric oxide (HgO) inside a glass tube, which liberated a gas he named "dephlogisticated air". He noted that candles burned brighter in the gas and that a mouse was more active and lived longer while breathing it. After breathing the gas himself, he wrote: "The feeling of it to my lungs was not sensibly different from that of common air, but I fancied that my breast felt peculiarly light and easy for some time afterwards." Priestley published his findings in 1775 in a paper titled "An Account of Further Discoveries in Air" which was included in the second volume of his book titled Experiments and Observations on Different Kinds of Air. Because he published his findings first, Priestley is usually given priority in the discovery.

## S16

**Question:** How many tons of dust remains in the air?

**Answer:** 132 million tons of dust remain in the air [a10p004:0:c76adad9].

**Passage `a10p004:0:c76adad9`** (Amazon rainforest)

> NASA's CALIPSO satellite has measured the amount of dust transported by wind from the Sahara to the Amazon: an average 182 million tons of dust are windblown out of the Sahara each year, at 15 degrees west longitude, across 1,600 miles (2,600 km) over the Atlantic Ocean (some dust falls into the Atlantic), then at 35 degrees West longitude at the eastern coast of South America, 27.7 million tons (15%) of dust fall over the Amazon basin, 132 million tons of dust remain in the air, 43 million tons of dust are windblown and falls on the Caribbean Sea, past 75 degrees west longitude.

## S17

**Question:** what is Internet2

**Answer:** Internet2 is a not-for-profit United States computer networking consortium led by members from the research and education communities, industry, and government [a13p020:0:5d2b009c].

**Passage `a13p020:0:5d2b009c`** (Packet switching)

> Internet2 is a not-for-profit United States computer networking consortium led by members from the research and education communities, industry, and government. The Internet2 community, in partnership with Qwest, built the first Internet2 Network, called Abilene, in 1998 and was a prime investor in the National LambdaRail (NLR) project. In 2006, Internet2 announced a partnership with Level 3 Communications to launch a brand new nationwide network, boosting its capacity from 10 Gbit/s to 100 Gbit/s. In October, 2007, Internet2 officially retired Abilene and now refers to its new, higher capacity network as the Internet2 Network.

## S18

**Question:** When was OPEC production of oil being surpassed?

**Answer:** OPEC's production was surpassed by other countries in 1981 [a08p023:0:0a1f87e2].

**Passage `a08p023:0:0a1f87e2`** (1973 oil crisis)

> OPEC soon lost its preeminent position, and in 1981, its production was surpassed by that of other countries. Additionally, its own member nations were divided. Saudi Arabia, trying to recover market share, increased production, pushing prices down, shrinking or eliminating profits for high-cost producers. The world price, which had peaked during the 1979 energy crisis at nearly $40 per barrel, decreased during the 1980s to less than $10 per barrel. Adjusted for inflation, oil briefly fell back to pre-1973 levels. This "sale" price was a windfall for oil-importing nations, both developing and developed.

## S19

**Question:** What were the years two Regulations that conflicted with an Italian law originate in the Simmenthal SpA case?

**Answer:** 1964 and 1968 [a09p011:3:dab38992]

**Passage `a09p011:3:dab38992`** (European Union law)

> hal SpA, a company, Simmenthal SpA, claimed that a public health inspection fee under an Italian law of 1970 for importing beef from France to Italy was contrary to two Regulations from 1964 and 1968. In "accordance with the principle of the precedence of Community law," said the Court of Justice, the "directly applicable measures of the institutions" (such as the Regulations in the case) "render automatically inapplicable any conflicting provision of current national law". This was necessary to prevent a "corresponding denial" of Treaty "obligations undertaken unconditionally and irrevocably by member states", that could "imperil the very foundations of the" EU. But despite the views of the Court of Justice, the national courts of member states have not accepted the same analysis.

## S20

**Question:** In addition to endowments and tuition, how do boarding schools cover their operating costs?

**Answer:** Fundraising drives [a19p025:0:51d820d1]

**Passage `a19p025:0:51d820d1`** (Private school)

> As of 2012, quality private schools in the United States charged substantial tuition, close to $40,000 annually for day schools in New York City, and nearly $50,000 for boarding schools. However, tuition did not cover operating expenses, particularly at boarding schools. The leading schools such as the Groton School had substantial endowments running to hundreds of millions of dollars supplemented by fundraising drives. Boarding schools with a reputation for quality in the United States have a student body drawn from throughout the country, indeed the globe, and a list of applicants which far exceeds their capacity.

## S21

**Question:** What is the name of Harvard's primary recreational sports facility?

**Answer:** The Malkin Athletic Center, known as the "MAC" [a20p025:0:5d088f0d]

**Passage `a20p025:0:5d088f0d`** (Harvard University)

> Harvard has several athletic facilities, such as the Lavietes Pavilion, a multi-purpose arena and home to the Harvard basketball teams. The Malkin Athletic Center, known as the "MAC", serves both as the university's primary recreation facility and as a satellite location for several varsity sports. The five-story building includes two cardio rooms, an Olympic-size swimming pool, a smaller pool for aquaerobics and other activities, a mezzanine, where all types of classes are held, an indoor cycling studio, three weight rooms, and a three-court gym floor to play basketball. The MAC offers personal trainers and specialty classes. It is home to Harvard volleyball, fencing and wrestling. The offices of several of the school's varsity coaches are also in the MAC.

## S22

**Question:** Who is the primary rival of the Harvard Crimson hockey team?

**Answer:** Cornell [a20p026:0:e378b2de]

**Passage `a20p026:0:e378b2de`** (Harvard University)

> Older than The Game by 23 years, the Harvard-Yale Regatta was the original source of the athletic rivalry between the two schools. It is held annually in June on the Thames River in eastern Connecticut. The Harvard crew is typically considered to be one of the top teams in the country in rowing. Today, Harvard fields top teams in several other sports, such as the Harvard Crimson men's ice hockey team (with a strong rivalry against Cornell), squash, and even recently won NCAA titles in Men's and Women's Fencing. Harvard also won the Intercollegiate Sailing Association National Championships in 2003.

## S23

**Question:** What system has an impact on income inequality?

**Answer:** A progressive tax system [a22p007:0:dfada8ac]

**Passage `a22p007:0:dfada8ac`** (Economic inequality)

> Another cause is the rate at which income is taxed coupled with the progressivity of the tax system. A progressive tax is a tax by which the tax rate increases as the taxable base amount increases. In a progressive tax system, the level of the top tax rate will often have a direct impact on the level of inequality within a society, either increasing it or decreasing it, provided that income does not change as a result of the change in tax regime. Additionally, steeper tax progressivity applied to social spending can result in a more equal distribution of income across the board. The difference between the Gini index for an income distribution before taxation and the Gini index after taxation is an indicator for the effects of such taxation.

## S24

**Question:** In what year did the scavenger hunt begin?

**Answer:** 1987 [a23p026:0:7d4a7573]

**Passage `a23p026:0:7d4a7573`** (University of Chicago)

> Every May since 1987, the University of Chicago has held the University of Chicago Scavenger Hunt, in which large teams of students compete to obtain notoriously esoteric items from a list. Since 1963, the Festival of the Arts (FOTA) takes over campus for 7–10 days of exhibitions and interactive artistic endeavors. Every January, the university holds a week-long winter festival, Kuviasungnerk/Kangeiko, which include early morning exercise routines and fitness workshops. The university also annually holds a summer carnival and concert called Summer Breeze that hosts outside musicians, and is home to Doc Films, a student film society founded in 1932 that screens films nightly at the university. Since 1946, the university has organized the Latke-Hamantash Debate, which involves humorous discussions about the relative merits and meanings of latkes and hamantashen.

## S25

**Question:** What process attributes new wealth to those that already have it?

**Answer:** Wealth concentration [a22p017:0:c89cd91b]

**Passage `a22p017:0:c89cd91b`** (Economic inequality)

> Wealth concentration is a theoretical[according to whom?] process by which, under certain conditions, newly created wealth concentrates in the possession of already-wealthy individuals or entities. According to this theory, those who already hold wealth have the means to invest in new sources of creating wealth or to otherwise leverage the accumulation of wealth, thus are the beneficiaries of the new wealth. Over time, wealth condensation can significantly contribute to the persistence of inequality within society. Thomas Piketty in his book Capital in the Twenty-First Century argues that the fundamental force for divergence is the usually greater return of capital (r) than economic growth (g), and that larger fortunes generate higher returns [pp. 384 Table 12.2, U.S. university endowment size vs. real annual rate of return]

## S26

**Question:** The adaptive immune system must distinguish between what types of molecules?

**Answer:** The adaptive immune system must distinguish between self and non-self molecules [a25p004:0:cca6a18e].

**Passage `a25p004:0:cca6a18e`** (Immune system)

> Both innate and adaptive immunity depend on the ability of the immune system to distinguish between self and non-self molecules. In immunology, self molecules are those components of an organism's body that can be distinguished from foreign substances by the immune system. Conversely, non-self molecules are those recognized as foreign molecules. One class of non-self molecules are called antigens (short for antibody generators) and are defined as substances that bind to specific immune receptors and elicit an immune response.

## S27

**Question:** What Senate committee did Singer speak to in July 2000?

**Answer:** United States Senate Committee on Commerce, Science and Transportation [a26p014:0:aca00480]

**Passage `a26p014:0:aca00480`** (Intergovernmental Panel on Climate Change)

> These studies were widely presented as demonstrating that the current warming period is exceptional in comparison to temperatures between 1000 and 1900, and the MBH99 based graph featured in publicity. Even at the draft stage, this finding was disputed by contrarians: in May 2000 Fred Singer's Science and Environmental Policy Project held a press event on Capitol Hill, Washington, D.C., featuring comments on the graph Wibjörn Karlén and Singer argued against the graph at a United States Senate Committee on Commerce, Science and Transportation hearing on 18 July 2000. Contrarian John Lawrence Daly featured a modified version of the IPCC 1990 schematic, which he mis-identified as appearing in the IPCC 1995 report, and argued that "Overturning its own previous view in the 1995 report, the IPCC presented the 'Hockey Stick' as the new orthodoxy with hardly an apology or explanation for the abrupt U-turn since its 1995 report".

## S28

**Question:** What is the name of another compelling continuation of the Fermat primality test?

**Answer:** The Baillie-PSW test is a compelling continuation of the Fermat primality test [a27p011:0:33aa9f3d].

**Passage `a27p011:0:33aa9f3d`** (Prime number)

> A particularly simple example of a probabilistic test is the Fermat primality test, which relies on the fact (Fermat's little theorem) that np≡n (mod p) for any n if p is a prime number. If we have a number b that we want to test for primality, then we work out nb (mod b) for a random value of n as our test. A flaw with this test is that there are some composite numbers (the Carmichael numbers) that satisfy the Fermat identity even though they are not prime, so the test has no way of distinguishing between prime numbers and Carmichael numbers. Carmichael numbers are substantially rarer than prime numbers, though, so this test can be useful for practical purposes. More powerful extensions of the Fermat primality test, such as the Baillie-PSW, Miller-Rabin, and Solovay-Strassen tests, are guaranteed to fail at least some of the time when applied to a composite number.

## S29

**Question:** Geoglyphs dating to what period were found in deforested land along the Amazon River?

**Answer:** 0–1250 [a10p006:0:a1fcce44]

**Passage `a10p006:0:a1fcce44`** (Amazon rainforest)

> The first European to travel the length of the Amazon River was Francisco de Orellana in 1542. The BBC's Unnatural Histories presents evidence that Orellana, rather than exaggerating his claims as previously thought, was correct in his observations that a complex civilization was flourishing along the Amazon in the 1540s. It is believed that the civilization was later devastated by the spread of diseases from Europe, such as smallpox. Since the 1970s, numerous geoglyphs have been discovered on deforested land dating between AD 0–1250, furthering claims about Pre-Columbian civilizations. Ondemar Dias is accredited with first discovering the geoglyphs in 1977 and Alceu Ranzi with furthering their discovery after flying over Acre. The BBC's Unnatural Histories presented evidence that the Amazon rainforest, rather than being a pristine wilderness, has been shaped by man for at least 11,000 years through practices such as forest gardening and terra preta.

## S30

**Question:** What is one part of the innate immune system that doesn't attack microbes directly?

**Answer:** Natural killer cells [a25p014:0:e5eb8f62]

**Passage `a25p014:0:e5eb8f62`** (Immune system)

> Natural killer cells, or NK cells, are a component of the innate immune system which does not directly attack invading microbes. Rather, NK cells destroy compromised host cells, such as tumor cells or virus-infected cells, recognizing such cells by a condition known as "missing self." This term describes cells with low levels of a cell-surface marker called MHC I (major histocompatibility complex) – a situation that can arise in viral infections of host cells. They were named "natural killer" because of the initial notion that they do not require activation in order to kill cells that are "missing self." For many years it was unclear how NK cells recognize tumor cells and infected cells. It is now known that the MHC makeup on the surface of those cells is altered and the NK cells become activated through recognition of "missing self". Normal body cells are not recognized and attacked by NK cells because they express intact self MHC antigens.

## S31

**Question:** What kind of T cells kill cells that are infected with pathogens?

**Answer:** Killer T cells [a25p017:0:1159e7c9].

**Passage `a25p017:0:1159e7c9`** (Immune system)

> Killer T cells are a sub-group of T cells that kill cells that are infected with viruses (and other pathogens), or are otherwise damaged or dysfunctional. As with B cells, each type of T cell recognizes a different antigen. Killer T cells are activated when their T cell receptor (TCR) binds to this specific antigen in a complex with the MHC Class I receptor of another cell. Recognition of this MHC:antigen complex is aided by a co-receptor on the T cell, called CD8. The T cell then travels throughout the body in search of cells where the MHC I receptors bear this antigen. When an activated T cell contacts such cells, it releases cytotoxins, such as perforin, which form pores in the target cell's plasma membrane, allowing ions, water and toxins to enter. The entry of another toxin called granulysin (a protease) induces the target cell to undergo apoptosis. T cell killing of host cells is particularly important in preventing the replication of viruses.

## S32

**Question:** Under normal conditions, T cells and antibodies produce what kind of peptides?

**Answer:** T cells produce peptides from self antigens presented by MHC molecules, while antibodies produced by B cells display peptides derived from foreign antigens on their surface MHC class II molecules [a25p020:0:52b61078].

**Passage `a25p020:0:52b61078`** (Immune system)

> A B cell identifies pathogens when antibodies on its surface bind to a specific foreign antigen. This antigen/antibody complex is taken up by the B cell and processed by proteolysis into peptides. The B cell then displays these antigenic peptides on its surface MHC class II molecules. This combination of MHC and antigen attracts a matching helper T cell, which releases lymphokines and activates the B cell. As the activated B cell then begins to divide, its offspring (plasma cells) secrete millions of copies of the antibody that recognizes this antigen. These antibodies circulate in blood plasma and lymph, bind to pathogens expressing the antigen and mark them for destruction by complement activation or for uptake and destruction by phagocytes. Antibodies can also neutralize challenges directly, by binding to bacterial toxins or by interfering with the receptors that viruses and bacteria use to infect cells.

## S33

**Question:** Who matters the most for economic growth?

**Answer:** The poor and the middle class matter the most for growth via a number of interrelated economic, social, and political channels [a22p028:0:c57d4faf].

**Passage `a22p028:0:c57d4faf`** (Economic inequality)

> According to International Monetary Fund economists, inequality in wealth and income is negatively correlated with the duration of economic growth spells (not the rate of growth). High levels of inequality prevent not just economic prosperity, but also the quality of a country's institutions and high levels of education. According to IMF staff economists, "if the income share of the top 20 percent (the rich) increases, then GDP growth actually declines over the medium term, suggesting that the benefits do not trickle down. In contrast, an increase in the income share of the bottom 20 percent (the poor) is associated with higher GDP growth. The poor and the middle class matter the most for growth via a number of interrelated economic, social, and political channels."

## S34

**Question:** What is the immune system of the brained known as?

**Answer:** The neuroimmune system [a25p000:0:70113353]

**Passage `a25p000:0:70113353`** (Immune system)

> The immune system is a system of many biological structures and processes within an organism that protects against disease. To function properly, an immune system must detect a wide variety of agents, known as pathogens, from viruses to parasitic worms, and distinguish them from the organism's own healthy tissue. In many species, the immune system can be classified into subsystems, such as the innate immune system versus the adaptive immune system, or humoral immunity versus cell-mediated immunity. In humans, the blood–brain barrier, blood–cerebrospinal fluid barrier, and similar fluid–brain barriers separate the peripheral immune system from the neuroimmune system which protects the brain.

## S35

**Question:** What was Sadat seeking by releasing Islamists from prison?

**Answer:** Sadat released Islamists from prison in tacit exchange for political support in his struggle against leftists [a30p006:0:92356785].

**Passage `a30p006:0:92356785`** (Islamism)

> Egyptian President Anwar Sadat – whose policies included opening Egypt to Western investment (infitah); transferring Egypt's allegiance from the Soviet Union to the United States; and making peace with Israel – released Islamists from prison and welcomed home exiles in tacit exchange for political support in his struggle against leftists. His "encouraging of the emergence of the Islamist movement" was said to have been "imitated by many other Muslim leaders in the years that followed."  This "gentlemen's agreement" between Sadat and Islamists broke down in 1975 but not before Islamists came to completely dominate university student unions. Sadat was later assassinated and a formidable insurgency was formed in Egypt in the 1990s. The French government has also been reported to have promoted Islamist preachers "in the hope of channeling Muslim energies into zones of piety and charity."

## S36

**Question:** When did the Rhine become borders with Francia?

**Answer:** The Rhine became a border between Middle and Western Francia in the 9th century [a28p039:0:23e99401].

**Passage `a28p039:0:23e99401`** (Rhine)

> By the 6th century, the Rhine was within the borders of Francia. In the 9th, it formed part of the border between Middle and Western Francia, but in the 10th century, it was fully within the Holy Roman Empire, flowing through Swabia, Franconia and Lower Lorraine. The mouths of the Rhine, in the county of Holland, fell to the Burgundian Netherlands in the 15th century; Holland remained contentious territory throughout the European wars of religion and the eventual collapse of the Holy Roman Empire, when the length of the Rhine fell to the First French Empire and its client states. The Alsace on the left banks of the Upper Rhine was sold to Burgundy by Archduke Sigismund of Austria in 1469 and eventually fell to France in the Thirty Years' War. The numerous historic castles in Rhineland-Palatinate attest to the importance of the river as a commercial route.

## S37

**Question:** Friedrich Ratzel thought what was needed for a state to survive?

**Answer:** Expansion [a31p007:0:2c12dbd0]

**Passage `a31p007:0:2c12dbd0`** (Imperialism)

> The Royal Geographical Society of London and other geographical societies in Europe had great influence and were able to fund travelers who would come back with tales of their discoveries. These societies also served as a space for travellers to share these stories.Political geographers such as Friedrich Ratzel of Germany and Halford Mackinder of Britain also supported imperialism. Ratzel believed expansion was necessary for a state’s survival while Mackinder supported Britain’s imperial expansion; these two arguments dominated the discipline for decades.

## S38

**Question:** Who was the first Warsaw President?

**Answer:** Jan Andrzej Menich [a32p030:0:0dbb1a3b]

**Passage `a32p030:0:0dbb1a3b`** (Warsaw)

> The mayor of Warsaw is called President. Generally, in Poland, the mayors of bigger cities are called presidents – i.e. such cities, which have over 100,000 people or these, where already was president before 1990. The first Warsaw President was Jan Andrzej Menich (1695–1696). Between 1975 and 1990 the Warsaw Presidents was simultaneously the Warsaw Voivode. Since 1990 the President of Warsaw had been elected by the City council. In the years of 1994–1999 the mayor of the district Centrum automatically was designated as the President of Warsaw: the mayor of Centrum was elected by the district council of Centrum and the council was elected only by the Centrum residents. Since 2002 the President of Warsaw is elected by all of the citizens of Warsaw.

## S39

**Question:** How many rooms does the Warsaw Historical Museum have?

**Answer:** 60 rooms [a32p042:0:84eae7c7]

**Passage `a32p042:0:84eae7c7`** (Warsaw)

> A fine tribute to the fall of Warsaw and history of Poland can be found in the Warsaw Uprising Museum and in the Katyń Museum which preserves the memory of the crime. The Warsaw Uprising Museum also operates a rare preserved and operating historic stereoscopic theatre, the Warsaw Fotoplastikon. The Museum of Independence preserves patriotic and political objects connected with Poland's struggles for independence. Dating back to 1936 Warsaw Historical Museum contains 60 rooms which host a permanent exhibition of the history of Warsaw from its origins until today.

## S40

**Question:** How much British military was in North America at start of War?

**Answer:** Few British troops were stationed in North America at the start of the French and Indian War [a33p011:0:20b21f32].

**Passage `a33p011:0:20b21f32`** (French and Indian War)

> At the start of the war, no French regular army troops were stationed in North America, and few British troops. New France was defended by about 3,000 troupes de la marine, companies of colonial regulars (some of whom had significant woodland combat experience). The colonial government recruited militia support when needed. Most British colonies mustered local militia companies, generally ill trained and available only for short periods, to deal with native threats, but did not have any standing forces.

## S41

**Question:** What seldom used term of a unit of force equal to 1000 pound s of force?

**Answer:** kip [a34p043:0:cd8f0ffe]

**Passage `a34p043:0:cd8f0ffe`** (Force)

> The pound-force has a metric counterpart, less commonly used than the newton: the kilogram-force (kgf) (sometimes kilopond), is the force exerted by standard gravity on one kilogram of mass. The kilogram-force leads to an alternate, but rarely used unit of mass: the metric slug (sometimes mug or hyl) is that mass that accelerates at 1 m·s−2 when subjected to a force of 1 kgf. The kilogram-force is not a part of the modern SI system, and is generally deprecated; however it still sees use for some purposes as expressing aircraft weight, jet thrust, bicycle spoke tension, torque wrench settings and engine output torque. Other arcane units of force include the sthène, which is equivalent to 1000 N, and the kip, which is equivalent to 1000 lbf.

## S42

**Question:** Only small portions of the Amazon forest are a result of what?

**Answer:** Naturally occurring [a10p000:0:5c929033]

**Passage `a10p000:0:5c929033`** (Amazon rainforest)

> The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist broadleaf forest that covers most of the Amazon basin of South America. This basin encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes territory belonging to nine nations. The majority of the forest is contained within Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana. States or departments in four nations contain "Amazonas" in their names.

## S43

**Question:** What was formed after the undeformed units were deposited?

**Answer:** After the undeformed units were deposited, no further deformation occurred, as stated in [a15p013:1:70c36fcd].

**Passage `a15p013:1:70c36fcd`** (Geology)

> stern Canada, the oldest known rock in the world have been metamorphosed to the point where their origin is undiscernable without laboratory analysis. In addition, these processes can occur in stages. In many places, the Grand Canyon in the southwestern United States being a very visible example, the lower rock units were metamorphosed and deformed, and then deformation ended and the upper, undeformed units were deposited. Although any amount of rock emplacement and rock deformation can occur, and they can occur any number of times, these concepts provide a guide to understanding the geological history of an area.

## S44

**Question:** What can not be mostly predicted beforehand?

**Answer:** Errant members who vote against the party line may face consequences such as deselection as official party candidates during future elections or expulsion from their parties [a29p016:0:3c6cc03b].

**Passage `a29p016:0:3c6cc03b`** (Scottish Parliament)

> The outcome of most votes can be predicted beforehand since political parties normally instruct members which way to vote. Parties entrust some MSPs, known as whips, with the task of ensuring that party members vote according to the party line. MSPs do not tend to vote against such instructions, since those who do are unlikely to reach higher political ranks in their parties. Errant members can be deselected as official party candidates during future elections, and, in serious cases, may be expelled from their parties outright. Thus, as with many Parliaments, the independence of Members of the Scottish Parliament tends to be low, and backbench rebellions by members who are discontent with their party's policies are rare. In some circumstances, however, parties announce "free votes", which allows Members to vote as they please. This is typically done on moral issues.

## S45

**Question:** When did Robert Mayow prove his theories?

**Answer:** 1679 [a07p002:0:8926d4df]

**Passage `a07p002:0:8926d4df`** (Oxygen)

> In the late 17th century, Robert Boyle proved that air is necessary for combustion. English chemist John Mayow (1641–1679) refined this work by showing that fire requires only a part of air that he called spiritus nitroaereus or just nitroaereus. In one experiment he found that placing either a mouse or a lit candle in a closed container over water caused the water to rise and replace one-fourteenth of the air's volume before extinguishing the subjects. From this he surmised that nitroaereus is consumed in both respiration and combustion.

## S46

**Question:** Which border does the Rhine flow from the north?

**Answer:** The Rhine flows from the north into Switzerland, specifically in the canton of Schaffhausen and parts of the cantons of Zürich and Basel-Stadt, before it turns west [a28p008:0:1be6243b, a28p009:0:c52af0cf].

**Passage `a28p008:0:1be6243b`** (Rhine)

> The Rhine emerges from Lake Constance, flows generally westward, as the Hochrhein, passes the Rhine Falls, and is joined by its major tributary, the river Aare. The Aare more than doubles the Rhine's water discharge, to an average of nearly 1,000 m3/s (35,000 cu ft/s), and provides more than a fifth of the discharge at the Dutch border. The Aare also contains the waters from the 4,274 m (14,022 ft) summit of Finsteraarhorn, the highest point of the Rhine basin. The Rhine roughly forms the German-Swiss border from Lake Constance with the exceptions of the canton of Schaffhausen and parts of the cantons of Zürich and Basel-Stadt, until it turns north at the so-called Rhine knee at Basel, leaving Switzerland.

**Passage `a28p009:0:c52af0cf`** (Rhine)

> In the centre of Basel, the first major city in the course of the stream, is located the "Rhine knee"; this is a major bend, where the overall direction of the Rhine changes from West to North. Here the High Rhine ends. Legally, the Central Bridge is the boundary between High and Upper Rhine. The river now flows North as Upper Rhine through the Upper Rhine Plain, which is about 300 km long and up to 40 km wide. The most important tributaries in this area are the Ill below of Strasbourg, the Neckar in Mannheim and the Main across from Mainz. In Mainz, the Rhine leaves the Upper Rhine Valley and flows through the Mainz Basin.

## S47

**Question:** In what year did Savery's engine pump in a mine?

**Answer:** 1712 [a06p001:0:0964b898]

**Passage `a06p001:0:0964b898`** (Steam engine)

> The first commercially successful true engine, in that it could generate power and transmit it to a machine, was the atmospheric engine, invented by Thomas Newcomen around 1712. It was an improvement over Savery's steam pump, using a piston as proposed by Papin. Newcomen's engine was relatively inefficient, and in most cases was used for pumping water. It worked by creating a partial vacuum by condensing steam under a piston within a cylinder. It was employed for draining mine workings at depths hitherto impossible, and also for providing a reusable water supply for driving waterwheels at factories sited away from a suitable "head". Water that had passed over the wheel was pumped back up into a storage reservoir above the wheel.

## S48

**Question:** Who was Kublai Khan's uncle?

**Answer:** Ögedei Khan [a24p003:0:7a1b030d]

**Passage `a24p003:0:7a1b030d`** (Yuan dynasty)

> Genghis Khan united the Mongol and Turkic tribes of the steppes and became Great Khan in 1206. He and his successors expanded the Mongol empire across Asia. Under the reign of Genghis' third son, Ögedei Khan, the Mongols destroyed the weakened Jin dynasty in 1234, conquering most of northern China. Ögedei offered his nephew Kublai a position in Xingzhou, Hebei. Kublai was unable to read Chinese but had several Han Chinese teachers attached to him since his early years by his mother Sorghaghtani. He sought the counsel of Chinese Buddhist and Confucian advisers. Möngke Khan succeeded Ögedei's son, Güyük, as Great Khan in 1251. He granted his brother Kublai control over Mongol held territories in China. Kublai built schools for Confucian scholars, issued paper money, revived Chinese rituals, and endorsed policies that stimulated agricultural and commercial growth. He adopted as his capital city Kaiping in Inner Mongolia, later renamed Shangdu.

## S49

**Question:** Which tribes did Genghis Khan fight against?

**Answer:** Jin dynasty [a24p003:0:7a1b030d]

**Passage `a24p003:0:7a1b030d`** (Yuan dynasty)

> Genghis Khan united the Mongol and Turkic tribes of the steppes and became Great Khan in 1206. He and his successors expanded the Mongol empire across Asia. Under the reign of Genghis' third son, Ögedei Khan, the Mongols destroyed the weakened Jin dynasty in 1234, conquering most of northern China. Ögedei offered his nephew Kublai a position in Xingzhou, Hebei. Kublai was unable to read Chinese but had several Han Chinese teachers attached to him since his early years by his mother Sorghaghtani. He sought the counsel of Chinese Buddhist and Confucian advisers. Möngke Khan succeeded Ögedei's son, Güyük, as Great Khan in 1251. He granted his brother Kublai control over Mongol held territories in China. Kublai built schools for Confucian scholars, issued paper money, revived Chinese rituals, and endorsed policies that stimulated agricultural and commercial growth. He adopted as his capital city Kaiping in Inner Mongolia, later renamed Shangdu.

## S50

**Question:** When does a T cell identify pathogens?

**Answer:** A T cell identifies pathogens when its T cell receptor (TCR) binds to a specific antigen in a complex with the MHC Class I receptor of another cell [a25p017:0:1159e7c9].

**Passage `a25p017:0:1159e7c9`** (Immune system)

> Killer T cells are a sub-group of T cells that kill cells that are infected with viruses (and other pathogens), or are otherwise damaged or dysfunctional. As with B cells, each type of T cell recognizes a different antigen. Killer T cells are activated when their T cell receptor (TCR) binds to this specific antigen in a complex with the MHC Class I receptor of another cell. Recognition of this MHC:antigen complex is aided by a co-receptor on the T cell, called CD8. The T cell then travels throughout the body in search of cells where the MHC I receptors bear this antigen. When an activated T cell contacts such cells, it releases cytotoxins, such as perforin, which form pores in the target cell's plasma membrane, allowing ions, water and toxins to enter. The entry of another toxin called granulysin (a protease) induces the target cell to undergo apoptosis. T cell killing of host cells is particularly important in preventing the replication of viruses.
