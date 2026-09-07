from ml_predictor import predict_job_details


job_description = """ 
The Opportunity

Avantor is looking for Skill Builder/Developer (SB) who builds new data analytics models and develops the data processing code for Decision Intelligent applications. The role involves building ETL pipelines, designing data warehouse models, and writing optimized SQL scripts that power intelligent decision-making applications.

The ideal candidate will have a strong foundation in BI/DWH concepts, columnar databases, and SQL development, with the ability to work closely with Solution Architects to translate business requirements into technical solutions. This role requires proficiency in Prompt Engineering for AI-assisted development, along with strong API knowledge and database experience.

Major Job Duties And Responsibilities

Build data crawlers to extract data from source systems using proprietary ETL platforms.
Troubleshoot issues faced during the entire life cycle of a skill.
Design and build data warehouse models in columnar databases, including Facts, Dimensions, and Measures.
Develop data processing scripts using SQL and optimize complex sequences of SQL queries.
Work with Solution Architects to understand business requirements and implement them.
Perform data analysis for complex use cases using SQL; document technical specifications for Decision Intelligent applications.
Ensure operational controls are maintained and perform operating control self-assessments as required.

Qualifications

Bachelor’s degree in Computer Science, Information Technology, Engineering, or equivalent experience.
5+ years of relevant experience as Skill Builder
Experience in data engineering, BI development, or analytics in a team-based environment.
2+ years of experience working in Agile/Scrum environments.

Knowledge, Skills, And Abilities

Strong knowledge of BI domain, DWH concepts, and reporting.
Basic Unix knowledge.
Familiarity with workflows in programming concepts.
Experience using columnar databases.
Experience working directly with customers and presenting ideas effectively.
Excellent communication and presentation skills.
Proficiency in Prompt Engineering for AI-assisted development workflows and automation.
Strong API integration knowledge for solution design and data connectivity.
Strong analytical, troubleshooting, and problem-solving skills.
Good understanding of software development lifecycle, change control procedures, and production support best practices.
Ability to manage multiple priorities and work collaboratively in an Agile delivery environment.

Added Advantage / Preferred Skills

Applied Python programming.
Applied Java programming.
Applied ML skills and R programming experience.
Experience working with JSON and XML files.
Experience using APIs for data integration and connectivity.
Knowledge of SAP and related enterprise systems.
Knowledge of supply chain fundamentals and terminology.
Experience using AI-assisted development tools such as ChatGPT Codex and Claude Cowork.

Disclaimer

The above statements are intended to describe the general nature and level of work being performed by employees assigned to this classification. They are not intended to be construed as an exhaustive list of all responsibilities, duties and skills required of employees assigned to this position. Avantor is proud to be an equal opportunity employer.

Why Avantor?

Dare to go further in your career. Join our global team of 14,000+ associates whose passion for discovery and determination to overcome challenges relentlessly advances life-changing science.

The work we do changes people's lives for the better. It brings new patient treatments and therapies to market, giving a cancer survivor the chance to walk his daughter down the aisle. It enables medical devices that help a little boy hear his mom's voice for the first time. Outcomes such as these create unlimited opportunities for you to contribute your talents, learn new skills and grow your career at Avantor.

We are committed to helping you on this journey through our diverse, equitable and inclusive culture which includes learning experiences to support your career growth and success. At Avantor, dare to go further and see how the impact of your contributions set science in motion to create a better world. Apply today!

EEO Statement

We are an Equal Employment/Affirmative Action employer and VEVRAA Federal Contractor. We do not discriminate in hiring on the basis of sex, gender identity, sexual orientation, race, color, religious creed, national origin, physical or mental disability, protected Veteran status, or any other characteristic protected by federal, state/province, or local law.

If you need a reasonable accommodation for any part of the employment process, please contact us by email at recruiting@avantorsciences.com and let us know the nature of your request and your contact information. Requests for accommodation will be considered on a case-by-case basis. Please note that only inquiries concerning a request for reasonable accommodation will be responded to from this email address.

Privacy Policy

We will use the personal information that you have submitted to us in order to consider your application for the relevant role.

Your privacy is important to us. Please click here for our Privacy Policy which explains the purposes for which we will use your personal information and the ways in which we will handle and retain your information. It also explains the rights you have in relation to your information, and how to contact us with any queries or requests.

3rd Party Non-solicitation Policy

By submitting candidates without having been formally assigned on and contracted for a specific job requisition by Avantor, or by failing to comply with the Avantor recruitment process, you forfeit any fee on the submitted candidates, regardless of your usual terms and conditions. Avantor works with a preferred supplier list and will take the initiative to engage with recruitment agencies based on its needs and will not be accepting any form of solicitation
"""


result = predict_job_details(job_description)


print("================================")
print("ML MODEL TEST")
print("================================")

print("Predicted Degree:",
      result["predicted_degree"])

print("Predicted Specialization:",
      result["predicted_specialization"])
