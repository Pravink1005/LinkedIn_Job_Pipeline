from ml_predictor import predict_job_details


job_description = """ 
NACL is hiring for the Chennai location and is looking for candidates who can join immediately. Only immediate joiners will be considered. Experience and skills: 3+ years of experience in financial services analytics, preferably in banking, insurance, or investment management. Proficiency in SQL and Python/Scala/Java for large-scale data processing and analysis. Expertise with big data technologies, including Spark, Data Lake, Delta Lake, and Hive. Strong quantitative and problem-solving skills with the ability to translate complex data into actionable insights and the ability to effectively convey technical concepts to non-technical audiences. Ability to work independently and collaboratively in a fast-paced, dynamic environment. Professional certifications such as CFA, FRM, or CPA are a plus.

"""


result = predict_job_details(job_description)


print("================================")
print("ML MODEL TEST")
print("================================")

print("Predicted Degree:",
      result["predicted_degree"])

print("Predicted Specialization:",
      result["predicted_specialization"])
