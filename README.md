
# Network Security with Machine Learning

The  project is aimed at developing and implementing a comprehensive network security solution leveraging machine learning techniques. Network security is a critical aspect of modern technology, with cyber threats constantly evolving in sophistication. Traditional security measures are often insufficient to protect against advanced and emerging threats. This project seeks to enhance network security by harnessing the power of machine learning to detect and mitigate security breaches and vulnerabilities in real-time.




## Dataset

The NSS KDD dataset, an extension of the popular KDD Cup 1999 dataset, consists of network traffic data used for intrusion detection research. Here are some of its feature names with one-line explanations:

    1. duration: Length of the connection in seconds.

    2. protocol_type: Type of network protocol used (e.g., TCP, UDP, ICMP).

    3. service: Type of network service accessed (e.g., http, ftp, ssh).

    4. flag: Status of the connection (e.g., SF for "normal" or S0 for "suspicious").

    5. src_bytes: Number of bytes sent from source to destination.

    6. dst_bytes: Number of bytes sent from destination to source.

    7. land: Indicates if the connection is from/to the same host/port (1 for yes, 0 for no).

    8. wrong_fragment: Number of "wrong" fragments in the packet.

    9. urgent: Number of urgent packets.

    10. hot: Number of "hot" indicators (e.g., login attempts).

    11. num_failed_logins: Number of failed login attempts.

    12. logged_in: Indicates if the user is logged in (1 for yes, 0 for no).

    13. num_compromised: Number of compromised conditions.

    14. root_shell: Indicates if root shell access was obtained (1 for yes, 0 for no).

    15. su_attempted: Indicates if "su" command attempted (1 for yes, 0 for no).

    16. num_root: Number of root accesses.

    17. num_file_creations: Number of file creation operations.

    18. num_shells: Number of shell prompts.

    19. num_access_files: Number of operations on access control files.

    20. num_outbound_cmds: Number of outbound commands in an ftp session.

    21. is_host_login: Indicates if login is a host-based login (1 for yes, 0 for no).

    22. is_guest_login: Indicates if login is a guest login (1 for yes, 0 for no).

    23. count: Number of connections to the same host as the current connection.

    24. srv_count: Number of connections to the same service as the current connection.

    25. serror_rate: Percentage of connections that had "SYN" errors.

    26. srv_serror_rate: Percentage of connections with "SYN" errors to the same service.

    27. rerror_rate: Percentage of connections that had "REJ" errors.

    28. srv_rerror_rate: Percentage of connections with "REJ" errors to the same service.

    29. same_srv_rate: Percentage of connections to the same service.

    30. diff_srv_rate: Percentage of connections to different services.

    31. srv_diff_host_rate: Percentage of connections to different hosts among the same service.

    32. dst_host_count: Number of connections to the same destination host.

    33. dst_host_srv_count: Number of connections to the same destination service.

    34. dst_host_same_srv_rate: Percentage of connections to the same service on the destination host.

    35. dst_host_diff_srv_rate: Percentage of connections to different services on the destination host.

    36. dst_host_same_src_port_rate: Percentage of connections from the same source port.

    37. dst_host_srv_diff_host_rate: Percentage of connections to different hosts for the same service.

    38. dst_host_serror_rate: Percentage of connections that had "SYN" errors for the destination host.

    39. dst_host_srv_serror_rate: Percentage of connections with "SYN" errors for the destination service.

    40. dst_host_rerror_rate: Percentage of connections that had "REJ" errors for the destination host.

    41. dst_host_srv_rerror_rate: Percentage of connections with "REJ" errors for the destination service.
## Machine Learning

- RandomForestClassifier - The data was fitted with random forest and was able to achieve and accuracy of around 97% on the testing data.

- Logistic Regression - Logistic Regression was also fitted on the data and was also able to achieve accuracy of around 97% but was few decimals less than random forest classifier.
## Run Locally

Clone the project

```bash
  git clone https://github.com/kirpalsingh225/Network-Security-with-Machine-Learning
```

Go to the project directory

```bash
  cd Network Security with Machine Learning
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  python src/components/data_ingestion.py
```

