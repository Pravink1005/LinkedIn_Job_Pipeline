from ml_predictor import extract_experience_years, extract_skills, extract_skills_structured, predict_job_details


job_description = """ 

About the job
Job Title: Power Platform DeveloperLocation: India – Coimbatore 
About the Company:NovintiX is a fast-growing engineering and digital transformation companyserving top Life Sciences, MedTech, and Healthcare clients across the US, Ireland, andEurope. We specialise in delivering high-impact digital programs, engineering services,and talent solutions that accelerate innovation, efficiency, and compliance.
Key Responsibilities
Build and optimize applications using Power Apps and Power Automate.
Design and deliver interactive dashboards in Power BI and Tableau.
Manage and integrate databases including SQL Server, Dataverse, Azure SQL, AWS RDS/Redshift.
Develop ETL pipelines with Azure Data Factory and AWS Glue.
Deploy, configure, and manage solutions across Azure and AWS platforms.
Ensure governance, security, and compliance in all solutions.

Required Skills
Hands-on expertise in Power Platform (Apps, Automate, BI) and Tableau.
Strong SQL and Dataverse knowledge.
Experience with Azure services (Synapse, Fabric, Blob Storage) and AWS services (S3, Lambda, Redshift).
Exposure to Tableau 
Proficiency in API integrations and CI/CD pipelines.
Excellent analytical, problem-solving, and communication skills.

Good to Have:
Tools: Power Apps (Canvas & Model-Driven), Power Automate, Dataverse, Power BI, SharePoint, Microsoft Teams, and Azure DevOps.
Skills: Low-code application development, workflow automation, business process optimization, Dataverse data modeling, integrations using REST APIs & connectors, and solution deployment (ALM).
Programming Languages: Power Fx, JavaScript, C#, SQL, and HTML/CSS.


Requirements added by the job poster

• Bachelor's Degree

• 6+ years of work experience with Microsoft Azure

• 6+ years of work experience with Python (Programming Language)

• 6+ years of work experience with DAX

"""


result = predict_job_details(job_description)


print("================================")
print("ML MODEL TEST")
print("================================")

print("Predicted Degree:",
      result["predicted_degree"])

print("Predicted Specialization:",
      result["predicted_specialization"])

print("Extracted Skills:",
      extract_skills(job_description))

print("Skills by Category:",
      extract_skills_structured(job_description))

print("Experience (Min, Max):",
      extract_experience_years(job_description))
