class NamingConventions():

    def __init__(self):

        self.data_basename = "Data"

        self.n_rows_basename = "NRows"

        self.n_cols_basename = "NCols"

        self.dtype_basename = "DType"

    def DataName(self, 
            namespace: str, 
            basename: str):
        
        return namespace + "/" + \
            basename + "/" + \
            self.data_basename
    
    def nRowsName(self, 
            namespace: str, 
            basename: str):
        
        return namespace + "/" + \
            basename + "/" + \
            self.n_rows_basename
    
    def nColsName(self, 
            namespace: str, 
            basename: str):

        return namespace + "/" + \
            basename + "/" + \
            self.n_cols_basename
    
    def dTypeName(self, 
            namespace: str, 
            basename: str):

        return namespace + "/" + \
            basename + "/" + \
            self.dtype_basename

class StringArray:

    # conventional encoding an array of strings over ros topics

    def __init__(self, 
            delimiter=";"):
        
        self.delimiter = delimiter

    def encode(self, strings):

        """
        Encodes a list of strings into a single string using the delimiter.
        :param strings: List[str] - The list of strings to encode.
        :return: str - The encoded string.
        """
        return self.delimiter.join(strings)

    def decode(self, encoded_string):
        """
        Decodes a string back into a list of strings using the delimiter.
        :param encoded_string: str - The encoded string.
        :return: List[str] - The list of decoded strings.
        """
        return encoded_string.split(self.delimiter)


