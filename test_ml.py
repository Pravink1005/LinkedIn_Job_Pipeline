from ml_predictor import predict_job_details


job_description = """ 


About the job
ROLE

Architecting the Future

El Codamics is looking for a skilled Backend Developer to architect and build robust server-side applications. Join our mission to build high-performance APIs and cloud infrastructures for global enterprises.

The Mission

Develop RESTful APIs and microservices

Design and optimize database schemas

Implement security best practices

Integrate third-party services and APIs

Optimize backend performance

Collaborate with DevOps teams

Core Requirements

 Bachelor's Degree in CS
 2+ Years Experience
 Node.js, Python, or PHP

Elite Benefits

Remote Work Growth Budget Global Network

Ready to Apply?

Send your resume to careers@elcodamics.com or fill the form.

Apply Now
"""


result = predict_job_details(job_description)


print("================================")
print("ML MODEL TEST")
print("================================")

print("Predicted Degree:",
      result["predicted_degree"])

print("Predicted Specialization:",
      result["predicted_specialization"])
