from test_extraction import TestEntityExtraction
from extract_entities import ExtractBioEntities
from generate_human_labels import get_human_generated_labels

"""# Uncomment this section if you want to run all 6,524 faculty bios through entity extraction process
print("Entity Extraction on all Faculty Bios: Begin Run")
extraction_model = ExtractBioEntities()
names, emails = extraction_model.perform_extractions()
print("Entity Extraction on all Faculty Bios: End Run")"""

print("Test New Faculty Bio Entity Extraction Methodology: Begin Run")
test_extraction = TestEntityExtraction()
metrics = test_extraction.test_extraction_performance()
print("Test New Faculty Bio Entity Extraction Methodology: End Run")
print("Entity Extraction Metrics Comparison for Old & New Methodologies: ", metrics)

print("Entity Extraction on 50 random Faculty Bios: Begin Run")
extraction_model = ExtractBioEntities()
names, emails = extraction_model.perform_extractions(run_subset=50, output_folder="subset_results")
print("Entity Extraction on 50 random Faculty Bios: End Run")
print("Faculty Names: ", names)
print("Faculty Emails: ", emails)

print("Demo: User Manual Evaluation on 3 Faculty Bios")
bios_to_run, has_name, has_email = get_human_generated_labels(0, 3, demo=True)
print("Demo Ended. You can view your output in test_results/user_generated_labels.csv")
