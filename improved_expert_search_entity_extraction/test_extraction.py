import os
import codecs
import numpy as np

from extract_entities import ExtractBioEntities
from generate_human_labels import get_human_generated_labels


class TestEntityExtraction:
    def __init__(self, seed=0, run_subset=100):
        """Initialize variables for instance of test_entity_extraction class

        Args:
            seed (int, optional): positive integer indicating random seed to use. Defaults to 0.
            run_subset (int, optional): positive integer indicating number of results to retrieve. Defaults to 100.
        """
        self.new_entity_extraction = ExtractBioEntities()
        self.seed = seed
        self.run_subset = run_subset

    def test_extraction_performance(self):
        """Retrieves subset of new and old entity extraction results. Retrieves same subset of human-generated labels.
        Turns extraction results into 1s/0s entity was/was not found. Compares human-generated labels with old entity
        extraction results to calculate accuracy, precision, and recall values for OLD version of ExpertSearch entity
        extraction. Compares human-generated labels with new entity extraction results to calculate accuracy, precision,
        and recall values for NEW version of ExpertSearch entity extraction (this code).

        Returns:
            metrics_dict (dict): accuracy, precision, and recall values for old and new entity extraction results for both names and emails
        """
        # run main program to get entity extraction results
        extracted_names, extracted_emails = self.new_entity_extraction.perform_extractions(
            seed=self.seed, run_subset=self.run_subset, output_folder="test_results"
        )
        has_extracted_name = self.convert_extraction_to_label(extracted_names)
        has_extracted_email = self.convert_extraction_to_label(extracted_emails)

        # get labels from results of previous implementation of ExpertSearch entity extraction
        (
            prev_version_extracted_names,
            prev_version_extracted_emails,
        ) = self.get_prev_version_results()
        prev_version_has_name = self.convert_extraction_to_label(
            prev_version_extracted_names
        )
        prev_version_has_email = self.convert_extraction_to_label(
            prev_version_extracted_emails
        )

        # get human generated labels
        num_run, has_name, has_email = get_human_generated_labels(
            self.seed, self.run_subset
        )

        # get confusion matrix stats for names and emails
        new_name_tp, new_name_tn, new_name_fp, new_name_fn = self.calc_confusion_matrix(
            has_extracted_name, has_name
        )
        (
            new_email_tp,
            new_email_tn,
            new_email_fp,
            new_email_fn,
        ) = self.calc_confusion_matrix(has_extracted_email, has_email)
        old_name_tp, old_name_tn, old_name_fp, old_name_fn = self.calc_confusion_matrix(
            prev_version_has_name, has_name
        )
        (
            old_email_tp,
            old_email_tn,
            old_email_fp,
            old_email_fn,
        ) = self.calc_confusion_matrix(prev_version_has_email, has_email)

        # create dictionary with accuracy, precision, and recall metrics for names and emails
        metrics_dict = {"old_version_results": {}, "new_version_results": {}}
        metrics_dict["new_version_results"]["name_accuracy"] = self.calc_accuracy(
            new_name_tp, new_name_tn, new_name_fp, new_name_fn
        )
        metrics_dict["new_version_results"]["email_accuracy"] = self.calc_accuracy(
            new_email_tp, new_email_tn, new_email_fp, new_email_fn
        )
        metrics_dict["new_version_results"]["name_precision"] = self.calc_precision(
            new_name_tp, new_name_fp
        )
        metrics_dict["new_version_results"]["email_precision"] = self.calc_precision(
            new_email_tp, new_email_fp
        )
        metrics_dict["new_version_results"]["name_recall"] = self.calc_recall(
            new_name_tp, new_name_fn
        )
        metrics_dict["new_version_results"]["email_recall"] = self.calc_recall(
            new_email_tp, new_email_fn
        )

        metrics_dict["old_version_results"]["name_accuracy"] = self.calc_accuracy(
            old_name_tp, old_name_tn, old_name_fp, old_name_fn
        )
        metrics_dict["old_version_results"]["email_accuracy"] = self.calc_accuracy(
            old_email_tp, old_email_tn, old_email_fp, old_email_fn
        )
        metrics_dict["old_version_results"]["name_precision"] = self.calc_precision(
            old_name_tp, old_name_fp
        )
        metrics_dict["old_version_results"]["email_precision"] = self.calc_precision(
            old_email_tp, old_email_fp
        )
        metrics_dict["old_version_results"]["name_recall"] = self.calc_recall(
            old_name_tp, old_name_fn
        )
        metrics_dict["old_version_results"]["email_recall"] = self.calc_recall(
            old_email_tp, old_email_fn
        )

        return metrics_dict

    def calc_confusion_matrix(self, predicted_labels, truth_labels):
        """Calculates the integer value of each box in a confusion matrix - i.e., number of true positives,
        true negatives, false positives, and false negatives - given two lists of the same length with identical
        value options (1s and 0s).

        Args:
            predicted_labels (list): 1s/0s indicating item found/not found in faculty bio by automated program
            truth_labels (list): 1s/0s indicating item found/not found in faculty bio by human user

        Returns:
            tp (int): true positives
            tn (int): true negatives
            fp (int): false positives
            fn (int): false negatives
        """
        predicted_labels = np.array(predicted_labels)
        truth_labels = np.array(truth_labels)
        positive = 1
        negative = 0
        tp = np.sum(
            np.logical_and(predicted_labels == positive, truth_labels == positive)
        )
        tn = np.sum(
            np.logical_and(predicted_labels == negative, truth_labels == negative)
        )
        fp = np.sum(
            np.logical_and(predicted_labels == positive, truth_labels == negative)
        )
        fn = np.sum(
            np.logical_and(predicted_labels == negative, truth_labels == positive)
        )

        return tp, tn, fp, fn

    def calc_accuracy(self, tp, tn, fp, fn):
        """Calculates accuracy.

        Args:
            tp (int): true positives
            tn (int): true negatives
            fp (int): false positives
            fn (int): false negatives

        Returns:
            accuracy (float): accuracy value
        """
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        return accuracy

    def calc_precision(self, tp, fp):
        """Calculates precision.

        Args:
            tp (int): number of true positives
            fp (int): number of false positives

        Returns:
            precision (float): precision value
        """
        precision = tp / (tp + fp)

        return precision

    def calc_recall(self, tp, fn):
        """Calculates recall.

        Args:
            tp (int): number of true positives
            fn (int): number of false negatives

        Returns:
            recall (float): recall value
        """
        recall = tp / (tp + fn)

        return recall

    def convert_extraction_to_label(self, extracted_entities):
        """Turns list of extracted entities into 1s (entity WAS found, therefore string is NOT empty)
        and 0s (entity was NOT found, therefore string IS empty).

        Args:
            extracted_entities (list): emails or names extracted using new or old ExpertSearch entity extraction

        Returns:
            labels (list): 1s (indicating entity was found) and 0s (indicating entity was not found)
        """
        labels = [1 if len(x) > 0 else 0 for x in extracted_entities]

        return labels

    def get_prev_version_results(self):
        """Retrieves entity extraction results from previous version of ExpertSearch.
        Will only retrieve self.run_subset faculty bio results, selected randomly using self.seed.

        Returns:
            selected_prev_names (list): subset of names from previous ExpertSearch entity extraction results
            selected_prev_emails (list): subset of email addresses from previous ExpertSearch entity extraction results
        """
        # set random seed and select subset of faculty bio numbers to evaluate
        np.random.seed(self.seed)
        bios_to_gather = np.random.choice(6524, self.run_subset)

        # load names.txt file and index with bios_to_gather
        file_path = os.path.join(
            "data", "previous_version_extraction_results", "names.txt"
        )
        with codecs.open(file_path, encoding="utf-8", errors="ignore") as f:
            prev_names = f.read()
        prev_names_list = np.array(prev_names.split("\n"))
        selected_prev_names = prev_names_list[bios_to_gather]

        # load emails.txt file and index with bios_to_gather
        file_path = os.path.join(
            "data", "previous_version_extraction_results", "emails.txt"
        )
        with codecs.open(file_path, encoding="utf-8", errors="ignore") as f:
            prev_emails = f.read()
        prev_emails_list = np.array(prev_emails.split("\n"))
        selected_prev_emails = prev_emails_list[bios_to_gather]

        return selected_prev_names, selected_prev_emails
