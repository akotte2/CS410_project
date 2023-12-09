import os
import codecs
import numpy as np
from datetime import datetime
from flair.data import Sentence
from flair.nn import Classifier
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class ExtractBioEntities:
    def __init__(self):
        """Initialize variables for instance of extract_bio_entities class

        self.start_run (datetime): time program began running, used to calculate full runtime later
        self.tagger (SequenceTagger): Flair 'ner-large' sequence tagger model
        self.tlds (tuple): valid top-level domain names used for email address extraction
        """
        self.start_run = datetime.now()
        self.tagger = Classifier.load("ner-large")
        self.tlds = tuple(self.get_tlds())

    def get_tlds(self):
        """Turns text file of valid top-level domain names into list of top-level domain name strings,
        each preceeded by a '.', with leading/trailing whitespace removed, and in lowercase format. Also
        includes duplicate list of tlds with '.' at end of tld string in order to recognize email addresses
        that occur at the end of sentences.

        Text file with all top level domains created using https://data.iana.org/TLD/tlds-alpha-by-domain.txt
        top_level_domains.txt last updated 12/01/23

        Returns:
            tlds (list): list of valid top-level domain names in format ready for email address extraction
        """
        tlds = []
        with open("top_level_domains.txt") as tlds_file:
            for line in tlds_file:
                tld = "." + line
                tld = tld.strip().lower()
                tlds.append(tld)

        # add second list of tlds with '.' at end to recognize email addresses occuring at the end of sentences
        end_of_sentence_tlds = []
        for tld in tlds:
            end_of_sentence_tlds.append(tld + ".")

        tlds = tlds + end_of_sentence_tlds

        return tlds

    def replace_dots(self, bio):
        """Often, faculty members will purposefully obscure their email addresses to avoid spam. Their strategy often
        involves replacing the "." symbol in the domain of the email address with either the "." symbol or the spelled-out
        word "dot" surrounded by additional spacing and/or special characters. This function finds all instances of these
        variations in "." and removes special characters, additional spacing, and changes "." to "dot".

        Note that this function will not replace "dot" as a substring within words if it occurs. For example,
        "dotty@gmail [dot] com" will NOT be changed to ".ty@gmail.com", but rather "dotty@gmail.com", which is what we expect
        for an email address.

        Args:
                bio (str): text file of an individual faculty bio converted to string using UTF-8 encoding

        Returns:
                bio (str): faculty bio with all variations of "." in email address replaced
        """
        dots = [
            ' "dot" ',
            " (dot) ",
            " \{dot\} ",
            " [dot] ",
            " /dot/ ",
            " \\dot\\ ",
            " <dot> ",
            " _dot_ ",
            " [dot} ",
            " {dot] ",
            " -dot- ",
            ' "." ',
            " (.) ",
            " {.} ",
            " [.] ",
            " /./ ",
            " \\.\\ ",
            " <.> ",
            " _._ ",
            " [.} ",
            " {.] ",
            " -.- ",
        ]
        stripped_dots = [s.strip() for s in dots]
        str_to_replace = dots + stripped_dots + [" dot ", " ."]
        for i in str_to_replace:
            bio = bio.replace(i, ".")

        return bio

    def replace_ats(self, bio):
        """Often, faculty members will purposefully obscure their email addresses to avoid spam. Their strategy often
        involves replacing the "@" symbol in the email address with either the "@" symbol or the spelled-out word "at"
        surrounded by additional spacing and/or special characters. This function finds all instances of these variations
        in "@" and removes special characters, additional spacing, and changes "at" to "@".

        Note that while it will replace the word "at" with the "@" symbol and merge the words that occured before and after,
        this will not affect the email entity extraction because it primarily depends on the existence of a "." followed by a valid
        top-level domain, not the "@" symbol. Additionally, this function will not replace "at" as a substring within words
        if it occurs. For example, "kathryn -at- gmail.com" will NOT be changed to "k@hryn@gmail.com", but rather to
        "kathryn@gmail.com", which is what we expect for an email address.

        Args:
                bio (str): text file of an individual faculty bio converted to string using UTF-8 encoding

        Returns:
                bio (str): faculty bio with all variations of "@" in email address replaced
        """
        ats = [
            ' "at" ',
            " (at) ",
            " \{at\} ",
            " [at] ",
            " /at/ ",
            " \\at\\ ",
            " <at> ",
            " _at_ ",
            " [at} ",
            " {at] ",
            " -at- ",
            ' "@" ',
            " (@) ",
            " {@} ",
            " [@] ",
            " /@/ ",
            " \\@\\ ",
            " <@> ",
            " _@_ ",
            " [@} ",
            " {@] ",
            " -@- ",
        ]
        stripped_ats = [s.strip() for s in ats]
        str_to_replace = ats + stripped_ats + [" at ", " @ "]
        for i in str_to_replace:
            bio = bio.replace(i, "@")

        return bio

    def clean_bio_for_names(self, bio):
        """Perform data cleaning on raw bio text by removing all leading and trailing whitespace, and then converting all
        characters to lowercase. Returns bio text that is ready for faculty name extraction.

        Args:
            bio (str): text file of an individual faculty bio converted to string using UTF-8 encoding

        Returns:
            cleaned_bio_lst (list): list of cleaned word strings from individual faculty bio ready for name extraction
        """
        cleaned_bio = bio.strip().lower()
        cleaned_bio_lst = cleaned_bio.split()

        return cleaned_bio_lst

    def clean_bio_for_emails(self, bio):
        """Perform data cleaning on bio text raw bio text by removing all leading and trailing whitespace, and then converting all
        characters to lowercase. Calls replace_ats() and replace_dots(). Returns bio text that is ready for faculty email address
        extraction.

        Args:
            bio (str): text file of an individual faculty bio converted to string using UTF-8 encoding

        Returns:
            cleaned_bio (str): cleaned text file of an individual faculty bio ready for email address extraction
        """
        cleaned_bio = bio.strip().lower()
        cleaned_bio = self.replace_ats(cleaned_bio)
        cleaned_bio = self.replace_dots(cleaned_bio)

        return cleaned_bio

    def extract_names(self, bio_lst):
        """Sends intervals of list of words in faculty bio to find_name() for named entity recognition. Will stop once the
        first name has been found.

        Intervals are used to speed up the program. Assumes name could be multiple words long (maximum of 5 words) and might
        not be recognized properly if split across intervals. Implements a "look back"/overlap across intervals in case name
        would be split when using non-overlapping intervals.

        Args:
            bio_lst (list): list of words created from cleaned text of individual faculty bio

        Returns:
            name_found (str): if name in bio according to find_name(), returns string value of name; else returns empty string
        """
        name_found = ""  # empty strings evaluate to False
        start = 0
        interval_range = 20
        overlap = 4

        # while no name has been found and more words are left to evaluate, send intervals of bio to find_name function for NER
        while start < len(bio_lst) and not name_found:
            end = start + interval_range
            if end > len(bio_lst):
                end = len(bio_lst)
            name_found = self.find_name(bio_lst[start:end])
            start += interval_range - overlap

        return name_found

    def find_name(self, bio_subset):
        """Uses the Flair ner-large model to predict labels for each token in a subset of faculty member's bio. Returns
        the first token labeled as a person (aka faculty's name). If no token is labled as a person, returns empty string.

        Args:
            bio_subset (list): list of subset of words from cleaned faculty bio

        Returns:
            str: if name in bio according to Flair, returns string value of first name found; else returns empty string
        """
        # convert subset of words from faculty bio to single string joined by single spaces ' '
        bio_str = " ".join(bio_subset)

        # perform entity classification on faculty bio string
        ner_predictions = Sentence(bio_str)
        self.tagger.predict(ner_predictions)

        # return first instance of token labeled as a person entity
        for label in ner_predictions.get_labels():
            if label.value == "PER":
                return label.data_point.text

        # if no token is labeled as a person entity, return empty string, which evaluates to False
        return ""

    def clean_email(self, email_address):
        """Often, faculty members will purposefully obscure their email addresses to avoid spam. Their strategy sometimes
        includes wrapping the recipient portion of their email in special characters, such as (kathryn)@gmail.com, which
        invalidates the email address. This function checks for and removes special characters often found in these positions
        from the first and last position of the recipient name.
        Additionally, if the faculty member used multiple "@" symbols in a row to avoid spam, this function discards the
        additional instances of "@" by splitting on "@". Example: "kathryn@@@gmail.com" becomes "kathryn@gmail.com".

        Args:
            email_address (str): email address extracted from faculty bio using extract_emails() function

        Returns:
            cleaned_email_address: email address without multiple "@" symbols in a row or special characters wrapping recipient name
        """

        # split email address on "@" symbol to get recipient and domain names as separate entities
        email_address = email_address.split("@")
        recipient_name = email_address[0]
        domain_name = email_address[-1]
        char_to_replace = ["<", ">", "{", "}", "[", "]", "(", ")", '"']

        try:
            if recipient_name[0] in char_to_replace:
                recipient_name = recipient_name[1:]

            if recipient_name[-1] in char_to_replace:
                recipient_name = recipient_name[:-1]

            # re-join recipient name and domain name with "@" symbol in between
            cleaned_email_address = recipient_name + "@" + domain_name
        except:
            # example of error from bio no. 3033: recipient_name = ')', domain_name = 'wayne.edu'
            cleaned_email_address = ""

        return cleaned_email_address

    def clean_name_for_email(self, name):
        """When browsing the bios, it appears that some faculty members omit the "@" symbol from their email
        to avoid spam. This function removes spaces and periods from the name found by the extract_names()
        function associated with a particular faculty bio so it can be used in one of the conditions within
        the extract_emails() function. This methodology makes use of the trend that many recipient names of
        email addresses contain a subset of the email address owner's first, middle, and/or last name.

        Example from bio no. 53: "email: lavalle uiuc.edu"

        Args:
                name (str): name of faculty member extracted using extract_names() function

        Returns:
                cleaned_name (str): cleaned name for use in extract_emails() function
        """
        cleaned_name = name.replace(" ", "")
        cleaned_name = cleaned_name.replace(".", "")

        return cleaned_name

    def extract_emails(self, bio, name):
        """Splits faculty bio on all whitespace characters, creating list of words (tokens) in bio. Performs email address
        entity extraction finding the first instance where a token ends in a valid top-level domain name. Then uses
        a series of conditionals (example for each documented below) to determine if some combination of the token and
        the two prior tokens form an email address. Uses rules about the placement of the "@" symbol in email addresses.

        Returns only first instance of a found email address. If no email address is found, will return empty string.

        Args:
            bio (str): cleaned text of individual faculty bio
            name (str): name of faculty member extracted from bio using extract_names() function

        Returns:
            str: if email address in bio, returns string value of first email address found; else returns empty string
        """
        tokenized_bio = bio.split()
        name = self.clean_name_for_email(name)

        for i in range(2, len(tokenized_bio)):
            token = tokenized_bio[i]
            prev_token = tokenized_bio[i - 1]
            prev2_token = tokenized_bio[i - 2]

            res = str(token).endswith(self.tlds)

            if res == True:
                if "http" in token:
                    # if token is url, continue looking for email address
                    continue

                if prev_token == "@" or prev_token.startswith("@"):
                    # example: token = ".edu", prev_token = "@illinois", prev2_token = "akotte2"
                    # example: token = "illinois.edu", prev_token = "@", prev2_token = "akotte2"
                    return self.clean_email(prev2_token + prev_token + token)
                elif "@" in prev_token or token.startswith("@"):
                    # example: token = "illinois.edu", prev_token = "akotte2@"
                    # example: token = "@illinois.edu", prev_token = "akotte2"
                    return self.clean_email(prev_token + token)
                elif "@" in token:
                    # example: token = "akotte2@illinois.edu"
                    return self.clean_email(token)
                elif prev_token in name:
                    # example from bio no. 53: token = "uiuc.edu", prev_token = "lavalle"
                    return self.clean_email(prev_token + "@" + token)
                else:
                    continue

        return ""

    def get_file_paths(self, output_folder):
        """Gets extraction input folder name and output file names based on preferred output folder name.
        Compatible with all operating systems.

        Args:
                output_folder (str): name of folder to save extraction results, must exist as sub-dir in cwd

        Returns:
                bios_path (str): path to folder with faculty bio text files (extraction input)
                name_path (str): filepath to save extracted faculty names (extraction output)
                email_path (str): filepath to save extracted faculty email addresses (extraction output)
        """
        bios_path = os.path.join("data", "compiled_bios")
        name_path = os.path.join(output_folder, "NEW_names.txt")
        email_path = os.path.join(output_folder, "NEW_emails.txt")

        return bios_path, name_path, email_path

    def perform_extractions(self, seed=0, run_subset=False, output_folder="results"):
        """Performs name and email address entity extraction on the faculty bio text files found using the original
        ExpertSearch code base. This function runs an updated, more effective version of entity extraction compared
        to the entity extraction code from the original ExpertSearch code base.

        If run_subset is False, then all 6,524 faculty bio text files will be run. If run_subset is an integer, then
        only run_subset faculty bio text files will be run. This subset will be randomly selected according to seed.

        Args:
            seed (int, optional): positive integer indicating random seed to use. Defaults to 0.
            run_subset (int/bool, optional): positive integer indicating number of results to retrieve. Defaults to False.
            output_folder (str, optional): folder name to save entity extraction results, must be in cwd. Defaults to "results".

        Returns:
            names (list): faculty names extracted from faculty bios using new entity extraction methodology
            email_addresses (list): faculty email addresses extracted from faculty bios using new entity extraction methodology
        """
        # get input/output filepaths
        bios_path, name_path, email_path = self.get_file_paths(output_folder)

        # set random seed
        np.random.seed(seed)

        # determine which bios to perform extractions on
        total_bios = len(os.listdir(bios_path))
        if run_subset:
            bios_to_run = np.random.choice(total_bios, run_subset)
        else:
            bios_to_run = range(total_bios - 1)

        # initialize lists to store extracted entities
        names = []
        email_addresses = []

        # for each faculty bio text file, read in as string with UTF-8 encoding and perform entity extractions
        for i in bios_to_run:
            print("Faculty Bio ID: ", i)
            file_path = os.path.join(bios_path, str(i) + ".txt")
            with codecs.open(file_path, encoding="utf-8", errors="ignore") as f:
                bio = f.read()

            # run name extraction on cleaned bio instance
            cleaned_bio_lst = self.clean_bio_for_names(bio)
            name = self.extract_names(cleaned_bio_lst)
            names.append(name)

            # run email address extraction on cleaned bio instance
            cleaned_bio = self.clean_bio_for_emails(bio)
            email_address = self.extract_emails(cleaned_bio, name)
            email_addresses.append(email_address)

        self.save_extractions(name_path, names, email_path, email_addresses)

        end_run = datetime.now()
        print("Entity extraction runtime: ", str(end_run - self.start_run))

        return names, email_addresses

    def save_extractions(self, name_path, names, email_path, email_addresses):
        """Writes extracted faculty names and email addresses to individual text files in order of original
        faculty bios.

        Args:
                name_path (str): filepath to save extracted faculty names
                names (list): 1D list of extracted faculty names
                email_path (str): filepath to save extracted faculty email addresses
                email_addresses (list): 1D list of extracted faculty email addresses

        Returns:
                None
        """
        # write extracted names to text file in order of original bios
        with open(name_path, "w") as f:
            for name in names:
                f.write(name)
                f.write("\n")
            f.close()

        # write extracted email addresses to text file in order of original bios
        with open(email_path, "w") as f:
            for email_address in email_addresses:
                f.write(email_address)
                f.write("\n")
            f.close()

        return None
