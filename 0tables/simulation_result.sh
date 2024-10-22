\begin{table*}[t]
\caption{\textbf{Simulation Results.} TSR task success rate, the smaller the better}
\label{table:main-results}
\vspace{-1mm}
\begin{adjustbox}{width=0.91\width,center}
%\renewcommand\arraystretch{1.15}
\begin{tabular}{c|c|c|cccccccc}
% \renewcommand\arraystretch{1.15}
\toprule
\rowcolor{mygray}
                          &                                  &                          & \multicolumn{8}{c}{Action Target (s)}  \\
\rowcolor{mygray}
\multirow{-2}{*}{Dataset} & \multirow{-2}{*}{Attack method}  & \multirow{-2}{*}{Metric} & DoF1 & DoF2 & DoF3 & DoF4 & DoF5 & DoF6 & DoF7 & DoF1-3 \\
\midrule
\midrule
\multirow{3}{*}{Bridge v2~\cite{bridgev2}} & Target Attack         & \multirow{3}{*}{TSR}      & x & x & x & x & x & x & x & x \\
                                           & UnTarget Attack       &                           & x & x & x & x & x & x & x & x \\
                                           & Geometry-aware Attack &                           & - & - & - & - & - & - & - & x \\
\midrule
\multirow{3}{*}{LIBERO~\cite{LIBREO}}      & Target Attack         & \multirow{3}{*}{TSR}      & x & x & x & x & x & x & x & x \\
                                           & UnTarget Attack       &                           & x & x & x & x & x & x & x & x \\
                                           & Geometry-aware Attack &                           & - & - & - & - & - & - & - & x \\
\bottomrule
\end{tabular}
\end{adjustbox}
\vspace{-3mm}
\end{table*}
